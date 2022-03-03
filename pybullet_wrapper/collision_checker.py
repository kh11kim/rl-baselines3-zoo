from dataclasses import dataclass
from itertools import combinations, product, combinations_with_replacement
from .core import Bullet
import numpy as np
from typing import List

@dataclass
class CollisionCheckLink:
    body_name: str
    link_name: str
    body_uid: int
    link_uid: int
    parent_uid: int

class BulletCollisionChecker:
    def __init__(self, bullet: Bullet):
        self.bullet = bullet
        self.all_links = self.get_all_links()
        self.collision_check_list = self.parse_collision_check_list(self.all_links)
        #self.all_collision_check_pairs = self.get_collision_check_pairs()
        #self.collision_objects = self.get_collision_objects()
        self.n_check = 0

    def get_all_links(self):
        link_list = []
        for name, body_uid in self.bullet._bodies_idx.items():
            info = self.bullet.get_joint_info(name, all=True)
            #base link
            # link_list.append(CollisionCheckLink(
            #     body_name=name, 
            #     link_name=name+"_base",
            #     body_uid=body_uid,
            #     link_uid=-1,
            #     parent_uid=None
            # ))
            for joint_idx in info:
                link_list.append(CollisionCheckLink(
                    body_name=name, 
                    link_name=info[joint_idx]["link_name"].decode('utf8'),
                    body_uid=body_uid,
                    link_uid=joint_idx,
                    parent_uid=info[joint_idx]["parent_index"]
                ))
        return link_list

    def parse_collision_check_list(self, link_list: List[CollisionCheckLink]):
        check_list = []
        for link1, link2 in combinations(link_list, 2):
            is_both_base = link1.link_uid == link2.link_uid == -1
            is_same_object = link1.body_name == link2.body_name
            is_adjacent = is_same_object & \
                (link1.parent_uid == link2.link_uid) | (link2.parent_uid == link1.link_uid)
            is_both_finger = ("finger" in link1.link_name) & ("finger" in link1.link_name)
            if (not is_both_base) & \
                (not is_adjacent) & \
                (not is_both_finger):
                check_list.append((
                    (link1, link2) # check-tuple
                ))
        return check_list

    def reset(self):
        self.n_check = 0

    def is_collision(self):
        self.n_check += 1
        margin = 0.0
        distances = self.compute_distances(self.collision_check_list, 0)
        if (distances < margin).any():
            #idx = (np.array(distances < margin) == True)[0]
            #print(self.collision_check_list[idx][0].body_name, self.collision_check_list[idx][0].link_name)
            #print(self.collision_check_list[idx][1].body_name, self.collision_check_list[idx][1].link_name)
            return True
        return False

    def compute_distances(self, check_list, max_distance=1.0):
        distances = []
        for link1, link2 in check_list:
            closest_points = self.bullet.physics_client.getClosestPoints(
                bodyA=link1.body_uid,
                bodyB=link2.body_uid,
                distance=max_distance,
                linkIndexA=link1.link_uid,
                linkIndexB=link2.link_uid,
            )
            if len(closest_points) == 0:
                distances.append(np.inf)
            else:
                distances.append(np.min([pt[8] for pt in closest_points]))
        return np.array(distances)

    def get_collision_check_list_by_name(self, name1, name2):
        result = []
        for tuple in self.collision_check_list:
            body_name1 = tuple[0].body_name
            body_name2 = tuple[1].body_name
            if ((body_name1 == name1) & (body_name2 == name2)) | \
                ((body_name2 == name1) & (body_name1 == name2)):
                result.append(tuple)
        return result

    def debug_get_collision_pair(self):
        pass

    # def get_collision_objects(self):
    #     collision_objects = {}
    #     for name, body_uid in self.bullet._bodies_idx.items():
    #         links = [-1] #base link
    #         info = self.bullet.get_joint_info(name, all=True)
    #         links += [link_idx for link_idx in info]
    #         collision_objects[name] = CollisionObject(
    #             body_name=name,
    #             body_uid=body_uid,
    #             link_uids=links,
    #             info=info
    #         )
    #     return collision_objects

    # def get_all_links(self):
    #     link_list = []
    #     for name, body_uid in self.bullet._bodies_idx.items():
    #         info = self.bullet.get_joint_info(name, all=True)
    #         for link_uid in info:
    #             link_list.append(
    #                 IndexedCollisionObject(body_uid, link_uid)
    #             ) #(body_uid, link_uid)
    #     return link_list

    # def get_self_collision_check_pairs(self, name):
    #     body_uid = self.collision_objects[name].body_uid
    #     links = self.collision_objects[name].link_uids
    #     collision_check_pairs = []
    #     for link1, link2 in combinations(links, 2):
    #         is_adjacent = abs(link1 - link2) == 1
    #         if not is_adjacent:
    #             collision_check_pairs.append(
    #                 (CollisionCheckLink(body_uid, link1), 
    #                 CollisionCheckLink(body_uid, link2))
    #             )
    #     return collision_check_pairs

    # def get_collision_check_pairs(self, obj1, obj2):
    #     collision_check_pairs = []
    #     is_same_body = obj1.body_uid == obj2.body_uid
    #     if is_same_body:
    #         check_list = combinations(obj1.link_uids, 2)
    #     else:
    #         check_list = product(obj1.link_uids, obj2.link_uids)

    #     for link1, link2 in check_list:
    #         is_duplicate = (link1 > link2)
    #         is_parent = is_same_body & (link1 == obj2.info[link2]["parent_index"])
    #         is_child = is_same_body & (obj1.info[link1]["parent_index"] == link2)
    #         if (not is_duplicate) & (not is_parent|is_child):
    #             collision_check_pairs.append(
    #                 (CollisionCheckLink(obj1.body_uid, link1), 
    #                 CollisionCheckLink(obj2.body_uid, link2))
    #             )
    #     return collision_check_pairs

    # def get_all_collision_check_pairs(self):
    #     collision_check_pairs = []
    #     for pair in combinations(self.all_links, 2):
    #         is_same_body = pair[0].body_uid == pair[1].body_uid
    #         is_adjacent = abs(pair[0].link_uid - pair[1].link_uid) == 1
    #         if not is_same_body | ((is_same_body) & (not is_adjacent)):
    #             collision_check_pairs.append(pair)
    #     return collision_check_pairs
    
    # def check_collision_pairs(self, check_pairs):
    #     for obj1, obj2 in check_pairs:
    #         dist_info = self.bullet.physics_client.getClosestPoints(
    #             bodyA=obj1.body_uid,
    #             bodyB=obj2.body_uid,
    #             distance=0.0,
    #             linkIndexA=obj1.link_uid,
    #             linkIndexB=obj2.link_uid,
    #         )
    #         if len(dist_info) != 0:
    #             return True
    #     return False

    

