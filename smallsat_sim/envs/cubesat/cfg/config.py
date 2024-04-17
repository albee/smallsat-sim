class Body():
    def __init__(self, name, pos=[0,0,0], quat=[1,0,0,0]) -> None:
        self.name = name
        self.pos = pos
        self.quat = quat


class EnvConfig:
    """Environment configuration class. Contains dynamic vehicle information.
    """
    model = 'cubesat'
    compiler = {'convexhull': 'true'}
    visual = {'headlight': {'ambient': '0.5 0.5 0.5', 'specular': '0.5 0.5 0.5', 'diffuse': '0.5 0.5 0.5'}}
    class Bodies:
        num_bodies = 1
        bodies_list = []
        for i in range(num_bodies):
            bodies_list.append(Body(name=f'body{i}',pos=[i,0,10]))
