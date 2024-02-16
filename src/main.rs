// vrchat-eye-bone-placer/src/main.rs

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use argmin::{
    core::{CostFunction, Error, IterState, Problem, Solver, State, TerminationStatus},
    solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing},
};
use eframe::{App, Frame};
use egui::{vec2, Align, Button, CentralPanel, Context as EguiContext, DragValue, Layout};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use futures::Future;
use glam::Vec3A;
use instant::{Duration, Instant};
use obj::ObjData;
use rand::distributions::{Distribution, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rfd::AsyncFileDialog;
use serde::{Deserialize, Serialize};
use std::{
    mem,
    sync::mpsc::{self, Receiver, Sender},
};

const INITIAL_TEMPERATURE: f32 = 100.0;
const MAX_SCALE: f32 = 10.0;
const CHANGE_SCALE: f32 = 1.0;
const INSIDE_PENALTY: f32 = 2.0;
const TIME_SLICE: Duration = Duration::from_millis(1000 / 10);

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
    stall_best: u64,
    reannealing_best: u64,
    solver_state: EyeBoneSolverState,
    commonmark_cache: CommonMarkCache,
}

enum EyeBoneSolverState {
    Idle,
    Solved(Sphere),
    Solving {
        problem: Problem<EyeBonePlacementProblem>,
        solver: SimulatedAnnealing<f32, Xoshiro256PlusPlus>,
        iter_state: IterState<Sphere, (), (), (), f32>,
    },
    Failed(String),
}

#[derive(Debug)]
struct EyeBonePlacementProblem {
    positions: Vec<Vec3A>,
    aabb: (Vec3A, Vec3A),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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

            CommonMarkViewer::new("help").show(
                ui,
                &mut self.commonmark_cache,
                include_str!("../README.md"),
            );

            ui.allocate_ui_with_layout(
                vec2(ui.available_width(), 0.0),
                Layout::left_to_right(Align::Center).with_main_justify(true),
                |ui| {
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
                },
            );

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

            ui.collapsing("Advanced", |ui| {
                ui.label(
                    "The following fields can be used to adjust how long the approximation runs \
for. Higher values mean a potentially-better approximation but a longer run time.",
                );
                ui.horizontal(|ui| {
                    ui.label("Stop after no progress is made after ");
                    ui.add(DragValue::new(&mut self.stall_best).speed(1000.0));
                    ui.label(" iterations");
                });
                ui.horizontal(|ui| {
                    ui.label("Retry if no progress is made after ");
                    ui.add(DragValue::new(&mut self.reannealing_best).speed(100.0));
                    ui.label(" iterations");
                });
            });

            ui.allocate_ui_with_layout(
                vec2(ui.available_width(), 0.0),
                Layout::left_to_right(Align::Center).with_main_justify(true),
                |ui| {
                    if ui
                        .add_enabled(
                            self.obj.is_some()
                                && !matches!(self.solver_state, EyeBoneSolverState::Solving { .. }),
                            Button::new("Place Eye Bone"),
                        )
                        .clicked()
                    {
                        self.init_eye_bone_placer();
                    }
                },
            );

            self.solver_state.tick();

            match self.solver_state {
                EyeBoneSolverState::Idle => {}
                EyeBoneSolverState::Solving { ref iter_state, .. } => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(format!("Solving, iteration {}…", iter_state.iter));
                    });
                }
                EyeBoneSolverState::Solved(ref sphere) => {
                    ui.label(format!(
                        "Success: Place the eye bone at ({}, {}, {}) and give it length {}.",
                        sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius
                    ));
                }
                EyeBoneSolverState::Failed(ref msg) => {
                    ui.label(msg);
                }
            }

            ui.label(&self.result_str);
        });
    }
}

impl EyeBonePlacerApp {
    fn init_eye_bone_placer(&mut self) {
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

        let mut problem = Problem::new(EyeBonePlacementProblem { positions, aabb });
        let mut solver = SimulatedAnnealing::new(INITIAL_TEMPERATURE)
            .unwrap()
            .with_temp_func(SATempFunc::Boltzmann)
            .with_stall_best(self.stall_best)
            .with_reannealing_best(self.reannealing_best);

        let mut state = IterState::new();
        state = state.param(Sphere {
            center: (aabb.0 + aabb.1) * 0.5,
            radius: (aabb.1 - aabb.0).length() * 0.5,
        });
        let (iter_state, _) = solver
            .init(&mut problem, state)
            .expect("Failed to initialize solver");

        self.solver_state = EyeBoneSolverState::Solving {
            problem,
            solver,
            iter_state,
        };
    }
}

impl EyeBoneSolverState {
    fn tick(&mut self) {
        let EyeBoneSolverState::Solving {
            ref mut problem,
            ref mut solver,
            ref mut iter_state,
        } = *self
        else {
            return;
        };

        let start_time = Instant::now();

        while (Instant::now() - start_time) < TIME_SLICE && !iter_state.terminated() {
            match <SimulatedAnnealing<_, _> as Solver<
                EyeBonePlacementProblem,
                IterState<_, _, _, _, _>,
            >>::terminate_internal(solver, iter_state)
            {
                TerminationStatus::Terminated(why) => {
                    *iter_state = mem::take(iter_state).terminate_with(why)
                }
                TerminationStatus::NotTerminated => {}
            }
            if iter_state.terminated() {
                break;
            }

            let Ok((new_iter_state, _)) = solver.next_iter(problem, mem::take(iter_state)) else {
                *self = EyeBoneSolverState::Failed(self.error_msg());
                return;
            };

            *iter_state = new_iter_state;
            if iter_state.terminated() {
                break;
            }

            iter_state.update();
            iter_state.increment_iter();
        }

        if !iter_state.terminated() {
            return;
        }

        match iter_state.best_param {
            Some(ref sphere) => *self = EyeBoneSolverState::Solved(sphere.clone()),
            None => *self = EyeBoneSolverState::Failed(self.error_msg()),
        }
    }

    fn error_msg(&self) -> String {
        if let EyeBoneSolverState::Solving { ref iter_state, .. } = *self {
            format!("Failed to place eye bone: {:?}", iter_state)
        } else {
            String::new()
        }
    }
}

impl Anneal for EyeBonePlacementProblem {
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

impl CostFunction for EyeBonePlacementProblem {
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
            solver_state: EyeBoneSolverState::Idle,
            stall_best: STALL_BEST,
            reannealing_best: REANNEALING_BEST,
            commonmark_cache: CommonMarkCache::default(),
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
