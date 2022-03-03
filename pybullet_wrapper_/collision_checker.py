from dataclasses import dataclass
from itertools import combinations
from .core import Bullet

@dataclass
class IndexedCollisionObject:
    body_uid: int
    link_uid: int

class BulletCollisionChecker:
    def __init__(self, bullet: Bullet):
        self.bullet = bullet
        self.links = self.get_all_links()
        self.collision_check_pairs = self.get_collision_check_pairs()
        self.n_check = 0

    def reset(self):
        self.n_check = 0

    def get_all_links(self):
        link_list = []
        for name, body_uid in self.bullet._bodies_idx.items():
            info = self.bullet.get_joint_info(name, all=True)
            for link_uid in info:
                link_list.append(
                    IndexedCollisionObject(body_uid, link_uid)
                ) #(body_uid, link_uid)
        return link_list
    
    def get_collision_check_pairs(self):
        collision_check_pairs = []
        for pair in combinations(self.links, 2):
            is_same_body = pair[0].body_uid == pair[1].body_uid
            is_adjacent = abs(pair[0].link_uid - pair[1].link_uid) == 1
            if not is_same_body | ((is_same_body) & (not is_adjacent)):
                collision_check_pairs.append(pair)
        return collision_check_pairs
    
    def is_collision(self):
        self.n_check += 0
        for obj1, obj2 in self.collision_check_pairs:
            dist_info = self.bullet.physics_client.getClosestPoints(
                bodyA=obj1.body_uid,
                bodyB=obj2.body_uid,
                distance=0.0,
                linkIndexA=obj1.link_uid,
                linkIndexB=obj2.link_uid,
            )
            if len(dist_info) != 0:
                return True
        return False