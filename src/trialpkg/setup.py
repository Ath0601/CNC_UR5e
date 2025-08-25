from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'trialpkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.*')),
        (os.path.join('share', package_name, 'meshes', 'visual'), glob('meshes/visual/*.*')),
        (os.path.join('share', package_name, 'meshes', 'collision'), glob('meshes/collision/*.*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='atharva',
    maintainer_email='atharva@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cart = trialpkg.cartesian_trajectory_node:main',
            'parser = trialpkg.gcodeparser:main',
            'vision = trialpkg.minimumvis:main',
            'home = trialpkg.homepos:main',
            'coordtest = trialpkg.newcoord:main',
            'coord = trialpkg.coordinate_transform:main',
            'gcodetest = trialpkg.gcodepath:main',
            'getpos = trialpkg.gcodetry:main',
            'carttest = trialpkg.cartesianpath:main',
            'test = trialpkg.cartsimple:main',
            'cnctest = trialpkg.gcodenew:main',
            'traj = trialpkg.minimaltraj:main'
        ],
    },
)
