pub mod gfx;
pub mod math;
pub mod topo;

use gfx::{Context, InterfaceConfig};
use ash::vk;

fn main() {
    env_logger::init();
    let render_cfg = InterfaceConfig::new("render")
        .require_transfer()
        .require_graphics();

    let ctxt = Context::builder("demo")
        .filter_device(|prop| prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        .with_interface(render_cfg)
        .build()
        .unwrap();
}
