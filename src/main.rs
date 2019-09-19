pub mod gfx;
pub mod math;
pub mod topo;

use gfx::Context;
use ash::vk;

fn main() {
    env_logger::init();
    let ctxt = Context::builder("demo")
        .filter_device(|prop| prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        .build()
        .unwrap();
}
