import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("franka_panda.xml")
data = mujoco.MjData(model)

view = mujoco.viewer.launch_passive(model, data)

        
while view.is_running():
    with view.lock():
        view.sync()