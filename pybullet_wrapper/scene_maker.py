import numpy as np
from typing import Any, Dict, Iterator, Optional
from .core import Bullet
import pybullet as p

class BulletSceneMaker:
    def __init__(self, bullet:Bullet):
        self.bullet = bullet

    def create_box(
        self,
        body_name: str,
        half_extents: np.ndarray,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = np.ones(4),
        specular_color: np.ndarray = np.zeros(3),
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        texture: Optional[str] = None,
    ) -> None:
        """Create a box.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            half_extents (np.ndarray): Half size of the box in meters, as (x, y, z).
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            texture (str or None, optional): Texture file name. Defaults to None.
        """
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}
        self._create_geometry(
            body_name,
            geom_type=self.bullet.physics_client.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        # if texture is not None:
        #     texture_path = os.path.join(get_data_path(), texture)
        #     texture_uid = self.physics_client.loadTexture(texture_path)
        #     self.physics_client.changeVisualShape(self.bullet._bodies_idx[body_name], -1, textureUniqueId=texture_uid)

    def create_cylinder(
        self,
        body_name: str,
        radius: float,
        height: float,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = np.zeros(4),
        specular_color: np.ndarray = np.zeros(3),
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a cylinder.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            height (float): The height in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        visual_kwargs = {
            "radius": radius,
            "length": height,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius, "height": height}
        self._create_geometry(
            body_name,
            geom_type=self.bullet.physics_client.GEOM_CYLINDER,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_sphere(
        self,
        body_name: str,
        radius: float,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = np.zeros(4),
        specular_color: np.ndarray = np.zeros(3),
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a sphere.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        visual_kwargs = {
            "radius": radius,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius}
        self._create_geometry(
            body_name,
            geom_type=self.bullet.physics_client.GEOM_SPHERE,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def _create_geometry(
        self,
        body_name: str,
        geom_type: int,
        mass: float = 0.0,
        position: np.ndarray = np.zeros(3),
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        visual_kwargs: Dict[str, Any] = {},
        collision_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        baseVisualShapeIndex = self.bullet.physics_client.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = self.bullet.physics_client.createCollisionShape(geom_type, **collision_kwargs)
            self.bullet._bodies_idx[body_name] = self.bullet.physics_client.createMultiBody(
                baseVisualShapeIndex=baseVisualShapeIndex,
                baseCollisionShapeIndex=baseCollisionShapeIndex,
                baseMass=mass,
                basePosition=position,
            )
        else:
            baseCollisionShapeIndex = -1
            self.bullet._ghost_idx[body_name] = self.bullet.physics_client.createMultiBody(
                baseVisualShapeIndex=baseVisualShapeIndex,
                baseCollisionShapeIndex=baseCollisionShapeIndex,
                baseMass=mass,
                basePosition=position,
            )

        if lateral_friction is not None:
            self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
        if spinning_friction is not None:
            self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)

    def create_plane(self, z_offset: float) -> None:
        """Create a plane. (Actually, it is a thin box.)

        Args:
            z_offset (float): Offset of the plane.
        """
        self.create_box(
            body_name="plane",
            half_extents=np.array([3.0, 3.0, 0.01]),
            mass=0.0,
            position=np.array([0.0, 0.0, z_offset - 0.01]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.85, 0.85, 0.85, 1.0]),
        )

    def create_table(
        self,
        length: float,
        width: float,
        height: float,
        x_offset: float = 0.0,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a fixed table. Top is z=0, centered in y.

        Args:
            length (float): The length of the table (x direction).
            width (float): The width of the table (y direction)
            height (float): The height of the table.
            x_offset (float, optional): The offet in the x direction.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        self.create_box(
            body_name="table",
            half_extents=np.array([length, width, height]) / 2,
            mass=0.0,
            position=np.array([x_offset, 0.0, -height / 2]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.95, 0.95, 0.95, 1]),
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
        )
    
    def make_sphere_obstacle(self, name, position, rgb_color=[0.,0.,1.]):
        if not name in self.bullet._bodies_idx:
            self.create_sphere(
                body_name=name,
                radius=0.02,
                mass=0.0,
                position=position,
                rgba_color=[*rgb_color,0.3],
                ghost=False
            )
        else:
            self.bullet.set_base_pose(name, position, np.array([0.0, 0.0, 0.0, 1.0]))
    
    def view_position(self, name, position, rgb_color=[1.,0.,0.]):
        if not name in self.bullet._ghost_idx:
            self.create_sphere(
                body_name=name,
                radius=0.02,
                mass=0.0,
                ghost=True,
                position=np.zeros(3),
                rgba_color=np.array([*rgb_color, 0.3]),
            )    
        self.bullet.set_base_pose(name, position, np.array([0.0, 0.0, 0.0, 1.0]))
    
    def view_frame(self, name, pos, orn):
        if not name in self.bullet._ghost_idx:
            self.bullet._ghost_idx[name] = self._make_axes()
        x_orn = p.getQuaternionFromEuler([0., np.pi/2, 0])
        y_orn = p.getQuaternionFromEuler([-np.pi/2, 0, 0])
        z_orn = [0., 0., 0., 1.]
        axis_orn = [x_orn, y_orn, z_orn]
        for i, idx in enumerate(self.bullet._ghost_idx[name]):
            _, orn_ = p.multiplyTransforms([0,0,0], orn, [0,0,0], axis_orn[i])
            #(orientation@axis_orn[i]).to_qtn()
            #orn_ = [*orn_[1:], orn_[0]]
            self.bullet.physics_client.resetBasePositionAndOrientation(
                bodyUniqueId=idx, posObj=pos, ornObj=orn_
            )

    def _make_axes(
        self,
        length=0.05
    ):
        radius = length/12
        visualFramePosition = [0,0,length/2]
        r, g, b = np.eye(3)
        orns = [
            [0, 0.7071, 0, 0.7071],
            [-0.7071, 0, 0, 0.7071],
            [0,0,0,1]
        ]
        a = 0.9
        shape_ids = []
        for color in [r, g, b]:
            shape_ids.append(
                self.bullet.physics_client.createVisualShape(
                    shapeType=self.bullet.physics_client.GEOM_CYLINDER,
                    radius=radius,
                    length=length,
                    visualFramePosition=visualFramePosition,
                    rgbaColor=[*color, a],
                    specularColor=[0., 0., 0.]
                )
            )
        axes_id = []
        for orn, shape in zip(orns, shape_ids):
            axes_id.append(
                self.bullet.physics_client.createMultiBody(
                    baseVisualShapeIndex=shape,
                    baseCollisionShapeIndex=-1,
                    baseMass=0.,
                    basePosition=[0,0,0],
                    baseOrientation=orn
                )
            )
        return axes_id