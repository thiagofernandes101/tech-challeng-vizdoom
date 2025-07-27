from __future__ import annotations
from abc import ABC
from typing import Optional

class GameElement(ABC):
    def __init__(self, label):
        self.__id = label.object_id
        self.__label = label.object_name
        self.__pos_x = label.object_position_x
        self.__pos_y = label.object_position_y
        self.__pos_z = label.object_position_z
        self.__angle = label.object_angle

    @property
    def id(self) -> int:
        return self.__id

    @property
    def label(self) -> str:
        return self.__label
    
    @property
    def pos_x(self) -> str:
        return self.__pos_x
    
    @property
    def pos_y(self) -> str:
        return self.__pos_y
    
    @property
    def pos_z(self) -> str:
        return self.__pos_z
    
    @property
    def angle(self) -> str:
        return self.__angle
    
    def __eq__(self, value)-> bool:
        if not isinstance(value, GameElement):
            return NotImplemented
        return self.id == value.id

class Enemy(GameElement):
    pass

class Player(GameElement):
    pass

class Colectable(GameElement):
    pass

class Blood(GameElement):
    pass

class Targer(GameElement):
    pass

class ElementFactory:
    @staticmethod
    def create(label)-> Optional[GameElement]:
        if label.object_name in ('Zombieman', 'ShotgunGuy', 'HellKnight', 'MarineChainsawVzd', 'BaronBall', 'Demon', 'ChaingunGuy'):
            return Enemy(label)
        elif label.object_name == 'DoomPlayer':
            return Player(label)
        elif label.object_name in ('ArmorBonus','BlueArmor','TeleportFog', 'RocketLauncher','Chainsaw','PlasmaRifle','Chaingun','SuperShotgun','Shotgun','Stimpack','Medikit','HealthBonus', 'ClipBox','RocketBox', 'CellPack', 'Clip', 'ShellBox'):
            return Colectable(label)
        elif label.object_name == 'Blood':
            return Blood(label)
        elif label.object_name == 'GreenArmor':
            return Targer(label)
        else:
            return None
