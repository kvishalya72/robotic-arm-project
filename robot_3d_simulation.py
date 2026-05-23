import math
import time

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    p = None
    pybullet_data = None
    PYBULLET_AVAILABLE = False


class Robot3DSimulator:
    """A simple 3D robot simulator using PyBullet.

    This class loads a KUKA arm URDF and supports smooth joint movement,
    basic pick-and-place interaction, and a physics-based simulation loop.
    """

    def __init__(self):
        self.attached_constraint = None
        self.gripper_closed = False
        self.last_message = "Simulation ready"

        if not PYBULLET_AVAILABLE:
            self.last_message = (
                "PyBullet is not installed. "
                "Install pybullet in your Python environment and run again."
            )
            return

        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        self.end_effector_index = 6
        self.joint_indices = []
        self.joint_limits = {}
        self.target_positions = {}
        self._initialize_joints()
        self.pickable_id = self._create_pickable_object()
        self.attached_constraint = None
        self.gripper_closed = False
        self.last_message = "Simulation ready"

        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[0.3, 0, 0.25],
        )

    def _initialize_joints(self):
        if not PYBULLET_AVAILABLE:
            return
        for index in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, index)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(index)
                position = p.getJointState(self.robot_id, index)[0]
                lower = info[8]
                upper = info[9]
                if lower > upper:
                    lower, upper = -math.pi, math.pi
                self.joint_limits[index] = (lower, upper)
                self.target_positions[index] = position

    def _create_pickable_object(self):
        if not PYBULLET_AVAILABLE:
            return -1
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], rgbaColor=[0.8, 0.2, 0.2, 1])
        object_id = p.createMultiBody(
            baseMass=0.12,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0.55, 0, 0.05],
        )
        return object_id

    def _clamp_joint(self, index, value):
        lower, upper = self.joint_limits.get(index, (-math.pi, math.pi))
        return max(min(value, upper), lower)

    def _set_joint_target(self, index, delta):
        if index not in self.target_positions:
            return
        current = self.target_positions[index]
        target = self._clamp_joint(index, current + delta)
        self.target_positions[index] = target

    def _attach_if_close(self):
        if not PYBULLET_AVAILABLE:
            self.last_message = "PyBullet not available; cannot pick."
            return
        if self.attached_constraint is not None:
            self.last_message = "Already holding object"
            return

        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        effector_pos = list(link_state[0])
        object_pos, _ = p.getBasePositionAndOrientation(self.pickable_id)
        distance = math.dist(effector_pos, object_pos)

        if distance > 0.12:
            self.last_message = "Move closer to the object to pick"
            return

        self.attached_constraint = p.createConstraint(
            self.robot_id,
            self.end_effector_index,
            self.pickable_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.gripper_closed = True
        self.last_message = "Picked object"

    def _release_object(self):
        if not PYBULLET_AVAILABLE:
            self.last_message = "PyBullet not available; cannot place."
            return
        if self.attached_constraint is None:
            self.last_message = "Nothing to place"
            return

        p.removeConstraint(self.attached_constraint)
        self.attached_constraint = None
        self.gripper_closed = False
        self.last_message = "Placed object"

    def move_3d_arm(self, command):
        if not PYBULLET_AVAILABLE:
            self.last_message = (
                "PyBullet not installed. "
                "Install pybullet or run in a Python environment with it."
            )
            return

        normalized = str(command).strip().upper()

        if normalized in {"ARM_LEFT", "ROTATE_LEFT", "LEFT"}:
            self._set_joint_target(0, -0.18)
            self.last_message = "Rotate base left"
        elif normalized in {"ARM_RIGHT", "ROTATE_RIGHT", "RIGHT"}:
            self._set_joint_target(0, 0.18)
            self.last_message = "Rotate base right"
        elif normalized in {"ARM_UP", "UP"}:
            self._set_joint_target(1, -0.16)
            self._set_joint_target(2, 0.18)
            self.last_message = "Lift arm up"
        elif normalized in {"ARM_DOWN", "DOWN"}:
            self._set_joint_target(1, 0.16)
            self._set_joint_target(2, -0.18)
            self.last_message = "Lower arm down"
        elif normalized in {"PICK", "GRIP_CLOSE", "CLOSE"}:
            self._attach_if_close()
        elif normalized in {"PLACE", "GRIP_OPEN", "OPEN"}:
            self._release_object()
        elif normalized in {"STOP", "ARM_STOP"}:
            self.last_message = "Stop motion"
        else:
            self.last_message = f"Unhandled command: {command}"

        self.step_simulation(steps=3)

    def step_simulation(self, steps=3):
        if not PYBULLET_AVAILABLE:
            time.sleep(1.0 / 60.0)
            return

        joint_ids = list(self.target_positions.keys())
        target_positions = [self.target_positions[j] for j in joint_ids]
        p.setJointMotorControlArray(
            self.robot_id,
            joint_ids,
            p.POSITION_CONTROL,
            targetPositions=target_positions,
            positionGains=[0.5] * len(joint_ids),
            forces=[120] * len(joint_ids),
        )
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def shutdown(self):
        if PYBULLET_AVAILABLE and p.isConnected():
            p.disconnect()

    def get_status(self):
        return self.last_message


_simulator = Robot3DSimulator()


def move_3d_arm(command):
    _simulator.move_3d_arm(command)


def step_simulation():
    _simulator.step_simulation()


def shutdown():
    _simulator.shutdown()


def get_sim_status():
    return _simulator.get_status()