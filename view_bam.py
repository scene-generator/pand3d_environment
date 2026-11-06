# view_bam.py
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    WindowProperties,
    CollisionTraverser,
    CollisionNode,
    CollisionRay,
    CollisionHandlerQueue,
    BitMask32,
    TextNode,
    ClockObject,
    TextureAttrib,
)
import argparse
from pathlib import Path
from math import sin, cos, radians
import sys


class BamViewer(ShowBase):
    def __init__(self, bam_path):
        super().__init__()

        self.disableMouse()
        self.setFrameRateMeter(True)

        self.root = self.loader.loadModel(bam_path)
        if self.root.isEmpty():
            raise FileNotFoundError(f"Could not load BAM model: {bam_path}")
        self.root.reparentTo(self.render)

        # Camera state
        self.cam_yaw = 0.0
        self.cam_pitch = -20.0
        bounds = self.root.getTightBounds()
        if bounds:
            lower, upper = bounds
            self.scene_center = (lower + upper) * 0.5
            scene_size = upper - lower
            self.cam_distance = max(scene_size.length() * 1.25, 30.0)
            self.cam_pos = [
                self.scene_center.x,
                self.scene_center.y - self.cam_distance,
                self.scene_center.z + self.cam_distance * 0.35,
            ]
        else:
            self.scene_center = self.root.getPos(self.render)
            self.cam_distance = 30.0
            self.cam_pos = [self.scene_center.x, self.scene_center.y - 30.0, self.scene_center.z + 12.0]
        self.camera.setPos(*self.cam_pos)
        self.camera.lookAt(self.scene_center)
        self.cam_yaw = self.camera.getH()
        self.cam_pitch = self.camera.getP()

        # Movement state
        self.keys = {
            "w": False, "a": False, "s": False, "d": False,
            "q": False, "e": False, "shift": False
        }

        # Mouse look / orbit
        self.mouse_down = False
        self.last_mouse = None

        # Display toggles
        self.wireframe_on = False
        self.textures_on = True
        self.bounds_on = False

        self._setup_controls()
        self._setup_picking()

        self.taskMgr.add(self.update_camera_task, "update_camera_task")
        self.taskMgr.add(self.update_picking_task, "update_picking_task")

        # self.root.ls()
        self.root.analyze()

    def _setup_controls(self):
        # Movement
        for key in ["w", "a", "s", "d", "q", "e", "shift", "z", "x"]:
            self.accept(key, self._set_key, [key, True])
            self.accept(f"{key}-up", self._set_key, [key, False])

        # Toggles
        self.accept("f", self.toggle_wireframe)
        self.accept("t", self.toggle_textures)
        self.accept("b", self.toggle_bounds)
        # self.accept("l", self.dump_scene_graph)
        self.accept("mouse1", self._mouse_down, [True])
        self.accept("mouse1-up", self._mouse_down, [False])
        self.accept("z", self.zoom_in)
        self.accept("x", self.zoom_out)

        self.accept("escape", sys.exit)

        self._set_help_text()

    def _set_help_text(self):
        cm = self.addScreenText(0.02, 0.96, "F: wireframe   T: textures   B: bounds   Mouse drag: orbit   WASD: pan   Z/X: zoom")
        cm.setScale(0.05)

    def addScreenText(self, x, y, text):
        tn = TextNode("help")
        tn.setText(text)
        np = self.aspect2d.attachNewNode(tn)
        np.setScale(0.05)
        np.setPos(x, 0, y)
        return np

    def _set_key(self, key, value):
        self.keys[key] = value

    def toggle_wireframe(self):
        self.wireframe_on = not self.wireframe_on
        if self.wireframe_on:
            self.root.setRenderModeWireframe()
        else:
            self.root.clearRenderMode()

    def toggle_textures(self):
        self.textures_on = not self.textures_on
        if self.textures_on:
            self.root.clearAttrib(TextureAttrib)
        else:
            self.root.setTextureOff(1)

    def toggle_bounds(self):
        self.bounds_on = not self.bounds_on
        if self.bounds_on:
            self.root.showBounds()
        else:
            self.root.hideBounds()

    def dump_scene_graph(self):
        self.root.ls()

    def _mouse_down(self, down):
        self.mouse_down = down
        self.last_mouse = None
    
    def zoom_in(self):
        fov = self.camLens.getFov()
        self.camLens.setFov(
            max(5.0, fov.x * 0.8),
            max(5.0, fov.y * 0.8),
        )


    def zoom_out(self):
        fov = self.camLens.getFov()
        self.camLens.setFov(
            min(120.0, fov.x * 1.2),
            min(120.0, fov.y * 1.2),
        )

    def _setup_picking(self):
        self.picker_trav = CollisionTraverser()
        self.picker_queue = CollisionHandlerQueue()

        self.picker_ray = CollisionRay()
        picker_node = CollisionNode("pickerRay")
        picker_node.addSolid(self.picker_ray)
        picker_node.setFromCollideMask(BitMask32.bit(1))
        picker_node.setIntoCollideMask(BitMask32.allOff())

        self.picker_np = self.camera.attachNewNode(picker_node)
        self.picker_trav.addCollider(self.picker_np, self.picker_queue)

    def update_picking_task(self, task):
        if not self.mouseWatcherNode.hasMouse():
            return Task.cont

        if self.mouse_down and self.mouseWatcherNode.is_button_down("mouse1"):
            mpos = self.mouseWatcherNode.getMouse()
            self.picker_ray.setFromLens(self.camNode, mpos.getX(), mpos.getY())
            self.picker_trav.traverse(self.render)

            if self.picker_queue.getNumEntries() > 0:
                self.picker_queue.sortEntries()
                entry = self.picker_queue.getEntry(0)
                picked = entry.getIntoNodePath()
                print("Picked:", picked.getName())
                print("Path:", picked)

                # Walk up to the nearest named parent
                np = picked
                while not np.isEmpty():
                    name = np.getName()
                    if name and name != "render":
                        print("Top-ish node:", name)
                        break
                    np = np.getParent()

        return Task.cont

    def update_camera_task(self, task):
        dt = ClockObject.getGlobalClock().getDt()
        move_speed = 120.0 * dt * (3.0 if self.keys["shift"] else 1.0)

        # Mouse orbit
        if self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            if self.mouse_down:
                if self.last_mouse is not None:
                    dx = m.getX() - self.last_mouse[0]
                    dy = m.getY() - self.last_mouse[1]
                    self.cam_yaw -= dx * 180.0
                    self.cam_pitch = max(-89.0, min(-5.0, self.cam_pitch + dy * 180.0))
                self.last_mouse = (m.getX(), m.getY())
            else:
                self.last_mouse = None

        # Keyboard movement in camera frame
        heading = radians(self.cam_yaw)
        forward = (sin(heading), cos(heading))
        right = (cos(heading), -sin(heading))

        if self.keys["w"]:
            self.scene_center.x += forward[0] * move_speed
            self.scene_center.y += forward[1] * move_speed
        if self.keys["s"]:
            self.scene_center.x -= forward[0] * move_speed
            self.scene_center.y -= forward[1] * move_speed
        if self.keys["d"]:
            self.scene_center.x += right[0] * move_speed
            self.scene_center.y += right[1] * move_speed
        if self.keys["a"]:
            self.scene_center.x -= right[0] * move_speed
            self.scene_center.y -= right[1] * move_speed
        if self.keys["e"]:
            self.scene_center.z += move_speed
        if self.keys["q"]:
            self.scene_center.z -= move_speed

        horizontal_distance = self.cam_distance * cos(radians(self.cam_pitch))
        self.cam_pos = [
            self.scene_center.x - sin(heading) * horizontal_distance,
            self.scene_center.y - cos(heading) * horizontal_distance,
            self.scene_center.z - sin(radians(self.cam_pitch)) * self.cam_distance,
        ]
        self.camera.setPos(*self.cam_pos)
        self.camera.lookAt(self.scene_center)

        return Task.cont

def safe_path(path_like: str | Path, base_dir: str | Path | None = None) -> Path:
    s = str(path_like)

    # Fix common accidental escape corruption from bad string literals
    s = s.replace("\x08", "/b")   # backspace from "\b"
    s = s.replace("\\", "//")     # normalize Windows separators
    s = s.replace("\t", "/t")     # accidental tab from "\t"
    s = s.replace("\r", "/r")
    s = s.replace("\n", "/n")

    p = Path(s)
    if base_dir is not None and not p.is_absolute():
        p = Path(base_dir) / p

    return p.resolve()

def main() -> None:
    default_bam = Path(__file__).resolve().parent / "assets" / "Terrain" / "Generate" / "baltimore" / "baltimore.bam"
    parser = argparse.ArgumentParser(description="View a Panda3D BAM scene.")
    parser.add_argument("bam_file", nargs="?", type=Path, default=default_bam, help="BAM file to display")
    args = parser.parse_args()

    bam_file = args.bam_file.expanduser()
    if not bam_file.is_absolute():
        bam_file = Path(__file__).parent / bam_file
    bam_file = bam_file.expanduser().resolve()
    if not bam_file.is_file():
        parser.error(f"BAM file does not exist: {bam_file}")

    app = BamViewer(bam_file)
    app.run()


if __name__ == "__main__":
    main()