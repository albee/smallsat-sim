class Body():
    def __init__(self, name, pos='0 0 0', quat='1 0 0 0') -> None:
        self.name = name
        self.pos = pos
        self.quat = quat


class EnvConfig:
    def __init__(self,args) -> None:
        self.model = 'cubesat'
        self.compiler = {'convexhull': 'true'}
        self.visual = {'headlight': {'ambient': '0.5 0.5 0.5', 'specular': '0.5 0.5 0.5', 'diffuse': '0.5 0.5 0.5'}}
        self.num_bodies = args.num_bodies
    class Bodies:
        def __init__(self,num_bodies) -> None:
            self.num_bodies = num_bodies
            self.bodies_list = []
            for i in range(self.num_bodies):
                self.bodies_list.append(Body(name=f'body{i}',pos=f"{i} 0 10"))
