// vrchat-eye-bone-placer/src/main.rs

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use argmin::{
    core::{CostFunction, Error, Executor},
    solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing},
};
use eframe::{App, Frame};
use egui::{
    text::LayoutJob, Button, CentralPanel, Color32, Context as EguiContext, FontId, TextFormat,
};
use futures::Future;
use glam::Vec3A;
use obj::ObjData;
use rand::distributions::{Distribution, Uniform};
use rfd::AsyncFileDialog;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{self, Receiver, Sender};

const INITIAL_TEMPERATURE: f32 = 100.0;
const MAX_SCALE: f32 = 10.0;
const CHANGE_SCALE: f32 = 1.0;
const INSIDE_PENALTY: f32 = 2.0;

#[cfg(debug_assertions)]
const STALL_BEST: u64 = 100_000;
#[cfg(debug_assertions)]
const REANNEALING_BEST: u64 = 3_000;

#[cfg(not(debug_assertions))]
const STALL_BEST: u64 = 500_000;
#[cfg(not(debug_assertions))]
const REANNEALING_BEST: u64 = 15_000;

struct EyeBonePlacerApp {
    obj: Option<ObjData>,
    result_str: String,
    obj_tx: Sender<ObjData>,
    obj_rx: Receiver<ObjData>,
}

#[derive(Debug)]
struct Problem {
    positions: Vec<Vec3A>,
    aabb: (Vec3A, Vec3A),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Sphere {
    center: Vec3A,
    radius: f32,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use eframe::NativeOptions;

    pretty_env_logger::init();

    let options = NativeOptions::default();

    let _ = eframe::run_native(
        "VRChat Eye Bone Placer",
        options,
        Box::new(|_| Box::<EyeBonePlacerApp>::default()),
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::{WebLogger, WebOptions, WebRunner};
    use log::LevelFilter;

    WebLogger::init(LevelFilter::Debug).ok();

    let web_options = WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        WebRunner::new()
            .start(
                "the_canvas_id",
                web_options,
                Box::new(|_| Box::<EyeBonePlacerApp>::default()),
            )
            .await
            .unwrap();
    })
}

impl App for EyeBonePlacerApp {
    fn update(&mut self, ctx: &EguiContext, _: &mut Frame) {
        CentralPanel::default().show(ctx, |ui| {
            if let Ok(obj) = self.obj_rx.try_recv() {
                self.obj = Some(obj);
            }

            ui.heading("VRChat Eye Bone Placer");

            let mut help_text = LayoutJob::default();
            help_text.append(
                "VRChat assumes that eyeballs are spheres. But sometimes they aren't. \
This is where this tool comes in.\n\n",
                0.0,
                TextFormat::default(),
            );
            help_text.append(
                "To have a non-spherical eye in VRChat, make the eyeball a static \
sphere, and attach the iris to a bone. This bone needs to be placed in such a way that it glides \
over the eyeball as close as possible without going inside. Placing this is a pain, so this tool \
automates it.\n\n",
                0.0,
                TextFormat::default(),
            );
            help_text.append("Choose a mesh in ", 0.0, TextFormat::default());
            help_text.append(
                ".obj",
                0.0,
                TextFormat::simple(FontId::monospace(12.0), Color32::LIGHT_GRAY),
            );
            help_text.append(" format containing the ", 0.0, TextFormat::default());
            help_text.append(
                "visible part",
                0.0,
                TextFormat::simple(FontId::proportional(14.0), Color32::RED),
            );
            help_text.append(" of ", 0.0, TextFormat::default());
            help_text.append(
                "one",
                0.0,
                TextFormat::simple(FontId::proportional(14.0), Color32::RED),
            );
            help_text.append(
                " eyeball, and this tool will tell you where to place the bone. \
This only places one bone, so make sure to only supply a single eyeball. You may mirror the bone \
in your modeling software to produce the other one. Also, it helps if you delete parts of the \
mesh that aren't visible, so that they aren't included in the approximation. Finally, remember \
that Blender switches the Y and Z axes whenever exporting in ",
                0.0,
                TextFormat::default(),
            );
            help_text.append(
                ".obj",
                0.0,
                TextFormat::simple(FontId::monospace(12.0), Color32::LIGHT_GRAY),
            );
            help_text.append(
                " format, so you may have to switch them back when entering the result in \
Blender.\n\n",
                0.0,
                TextFormat::default(),
            );
            help_text.append(
                "This is essentially just approximating the surface of whatever you \
provide with a sphere.\n\n",
                0.0,
                TextFormat::default(),
            );
            help_text.append(
                "Note that it may take a few seconds to place the bone when you click \
\"Place Eye Bone\". This is normal.",
                0.0,
                TextFormat::default(),
            );
            ui.label(help_text);

            if ui.button("Load .obj…").clicked() {
                let obj_tx = self.obj_tx.clone();
                let task = AsyncFileDialog::new()
                    .add_filter("obj files", &["obj"])
                    .set_title("VRChat Eye Bone Placer")
                    .pick_file();
                execute(async move {
                    if let Some(file) = task.await {
                        let obj_data = file.read().await;
                        if let Ok(obj_data) = ObjData::load_buf(&*obj_data) {
                            let _ = obj_tx.send(obj_data);
                        }
                    }
                });
            }

            match self.obj {
                None => {
                    ui.small("No .obj loaded");
                }
                Some(ref obj) => {
                    ui.small(&format!(
                        "Loaded .obj, {} {}",
                        obj.position.len(),
                        if obj.position.len() == 1 {
                            "vertex"
                        } else {
                            "vertices"
                        }
                    ));
                }
            }

            if ui
                .add_enabled(self.obj.is_some(), Button::new("Place Eye Bone"))
                .clicked()
            {
                self.place_eye_bone();
            }

            ui.label(&self.result_str);
        });
    }
}

impl EyeBonePlacerApp {
    fn place_eye_bone(&mut self) {
        let positions: Vec<_> = self
            .obj
            .as_ref()
            .unwrap()
            .position
            .iter()
            .map(|&v| Vec3A::from_array(v))
            .filter(|v| v.x >= 0.0)
            .collect();

        let aabb = positions
            .iter()
            .fold((Vec3A::MAX, Vec3A::MIN), |acc, &elem| {
                (acc.0.min(elem), acc.1.max(elem))
            });

        let problem = Problem { positions, aabb };

        let solver = SimulatedAnnealing::new(INITIAL_TEMPERATURE)
            .unwrap()
            .with_temp_func(SATempFunc::Boltzmann)
            .with_stall_best(STALL_BEST)
            .with_reannealing_best(REANNEALING_BEST);

        let result = Executor::new(problem, solver)
            .configure(|anneal| {
                anneal.param(Sphere {
                    center: (aabb.0 + aabb.1) * 0.5,
                    radius: (aabb.1 - aabb.0).length() * 0.5,
                })
            })
            .run()
            .unwrap();

        match result.state().best_param {
            Some(ref sphere) => {
                self.result_str = format!(
                    "Success: Place the eye bone at ({}, {}, {}) and give it length {}.",
                    sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius
                );
            }
            None => {
                self.result_str = format!("Failed to place eye bone: {:?}", result.state);
            }
        }
    }
}

impl Anneal for Problem {
    type Param = Sphere;

    type Output = Sphere;

    type Float = f32;

    fn anneal(&self, param: &Self::Param, extent: Self::Float) -> Result<Self::Output, Error> {
        let mut rng = rand::thread_rng();
        let mut param = (*param).clone();

        let aabb_center = (self.aabb.0 + self.aabb.1) * 0.5;
        let aabb_half_dims = (self.aabb.1 - self.aabb.0) * 0.5;

        let extents = (self.aabb.1 - self.aabb.0) * extent / INITIAL_TEMPERATURE * CHANGE_SCALE;

        let max_radius = (self.aabb.1 - self.aabb.0).max_element();
        let radius_extent = max_radius * extent / INITIAL_TEMPERATURE * CHANGE_SCALE;

        param.center += Vec3A::new(
            Uniform::new(-extents.x, extents.x).sample(&mut rng),
            Uniform::new(-extents.y, extents.y).sample(&mut rng),
            Uniform::new(-extents.z, extents.z).sample(&mut rng),
        );
        param.center = param.center.clamp(
            aabb_center - aabb_half_dims * MAX_SCALE,
            aabb_center + aabb_half_dims * MAX_SCALE,
        );

        param.radius += Uniform::new(-radius_extent, radius_extent).sample(&mut rng);
        param.radius = param.radius.clamp(0.0, radius_extent * MAX_SCALE);

        Ok(param)
    }
}

impl CostFunction for Problem {
    type Param = Sphere;

    type Output = f32;

    // Sphere surface distance
    fn cost(&self, sphere: &Sphere) -> Result<Self::Output, Error> {
        Ok(self
            .positions
            .iter()
            .map(|&p| {
                let mut dist = (p - sphere.center).length() - sphere.radius;

                // Try not to go inside the eyeball…
                if dist < 0.0 {
                    dist *= INSIDE_PENALTY;
                }

                dist * dist
            })
            .sum())
    }
}

impl Default for EyeBonePlacerApp {
    fn default() -> Self {
        let (obj_tx, obj_rx) = mpsc::channel();
        Self {
            obj: Default::default(),
            result_str: Default::default(),
            obj_tx,
            obj_rx,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn execute<F>(future: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    use std::thread;

    thread::spawn(move || futures::executor::block_on(future));
}

#[cfg(target_arch = "wasm32")]
fn execute<F>(future: F)
where
    F: Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(future);
}
