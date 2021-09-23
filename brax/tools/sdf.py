import xml.etree.ElementTree as ET
from brax.physics import config_pb2
from typing import AnyStr, Optional
import brax



filename = '/home/ros2/Documents/gazebo-model-test/Untitled/model.sdf'

# with open (filename, "r") as myfile:
#     data=myfile.readlines()
#
# data1 = ''.join(data)
# #rint(f"data{data1}")   
# sdf_str = ET.fromstring(data1)
#print(f"data{data_str.link}") 



       
def get_all_model_names(sdf_str):
    models = []
    for model in sdf_str.findall('model'):
        models.append(model.attrib.get('name'))
    return models
    
    
def get_sdf_link_names_from_a_sdf_model(model_name, sdf_str):
    bodies = []
    for model in sdf_str.findall('model'):
        if model.attrib.get('name') == model_name:
            # print(f"Found model >{model.attrib}<")
            for link in model.findall('link'):
                # print(f"\tFound body name >{link.attrib.get('name')}<")
                bodies.append(link.attrib.get('name'))
                
    return bodies
            
            
def get_sdf_geometry_from_a_sdf_model_link(model_name, link_name, sdf_str):
    for model in sdf_str.findall('model'):
        if model.attrib.get('name') == model_name:
            # print(f"Found model >{model.attrib}<")
            for link in model.findall('link'):
                if link.attrib.get('name') == link_name:
                    for collision in link.findall('collision'):
                        # print(f"\t\tFound collision tag >{collision.tag}< collision attrib >{collision.attrib}<")
                        for geometry in collision.findall('geometry'):
                            # print(f"\t\t\tFound geometry tag >{geometry}< geometry text >{geometry.attrib}<")
                            for x in geometry.findall('*'):
                                # print(f"\t\t\t\tGeometry is tag >{x.tag}< x text >{x.attrib}<")
                                return x.tag
                
   
def get_sdf_box_size_from_a_sdf_model_link(model_name, link_name, geometry_name, sdf_str):
    for model in sdf_str.findall('model'):
        if model.attrib.get('name') == model_name:
            # print(f"Found model >{model.attrib}<")
            for link in model.findall('link'):
                if link.attrib.get('name') == link_name:
                    for collision in link.findall('collision'):
                        # print(f"\t\tFound collision tag >{collision.tag}< collision attrib >{collision.attrib}<")
                        for geometry in collision.findall('geometry'):
                            # print(f"\t\t\tFound geometry tag >{geometry}< geometry text >{geometry.attrib}<")
                            for x in geometry.findall('*'):
                                if x.tag == geometry_name:
                                    # print(f"\t\t\t\tGeometry is tag >{x.tag}< x text >{x.attrib}<")
                                    for y in x.findall('*'):
                                        # print(f"\t\t\t\t\t{x.tag} is tag >{y.tag}< y text >{y.text}<")
                                        return y.text
                                         
           
def get_sdf_mass_from_a_sdf_model_link(model_name, link_name, geometry_name, sdf_str):
    for model in sdf_str.findall('model'):
        if model.attrib.get('name') == model_name:
            # print(f"Found model >{model.attrib}<")
            for link in model.findall('link'):
                if link.attrib.get('name') == link_name:
                    # print(f"Found link_name >{link_name}<")
                    for inertial in link.findall('inertial'):
                        # print(f"inertial >{inertial}<")
                        for mass in inertial.findall('mass'):
                            # print(f"mass >{mass.text}<")
                            return mass.text
                    
            
            
def get_sdf_pose_from_a_sdf_model_link(model_name, link_name, geometry_name, sdf_str):
    for model in sdf_str.findall('model'):
        if model.attrib.get('name') == model_name:
            # print(f"Found model >{model.attrib}<")
            for link in model.findall('link'):
                if link.attrib.get('name') == link_name:
                    # print(f"Found link_name >{link_name}<")
                    for pose in link.findall('pose'):
                        # print(f"pose >{pose.text}<")
                        return pose.text
    


def get_sdf_joint_names_from_a_sdf_model(model_name, sdf_str):
    joints = []
    for model in sdf_str.findall('model'):
        if model.attrib.get('name') == model_name:
            # print(f"Found model >{model.attrib}<")
            for joint in model.findall('joint'):
                # print(f"\tFound body name >{link.attrib.get('name')}<")
                joints.append(joint.attrib.get('name'))
                
    return joints


def get_sdf_joint_parent_from_a_sdf_model(model_name, joint_name, sdf_str):
    for model in sdf_str.findall('model'):
        if model.attrib.get('name') == model_name:
            # print(f"Found model >{model.attrib}<")
            for joint in model.findall('joint'):
                if joint.attrib.get('name') == joint_name:
                    # print(f"\tFound body name >{link.attrib.get('name')}<")
                    for parent in joint.findall('parent'):
                        # print(f"\tFound parent name >{parent.text}<")
                        return parent.text
                    
                    
def get_sdf_joint_child_from_a_sdf_model(model_name, joint_name, sdf_str):
    for model in sdf_str.findall('model'):
        if model.attrib.get('name') == model_name:
            # print(f"Found model >{model.attrib}<")
            for joint in model.findall('joint'):
                if joint.attrib.get('name') == joint_name:
                    # print(f"\tFound body name >{link.attrib.get('name')}<")
                    for child in joint.findall('child'):
                        # print(f"\tFound parent name >{parent.text}<")
                        return child.text


def get_sdf_joint_origin_from_a_sdf_model(model_name, joint_name, sdf_str):
    for model in sdf_str.findall('model'):
        if model.attrib.get('name') == model_name:
            # print(f"Found model >{model.attrib}<")
            for joint in model.findall('joint'):
                if joint.attrib.get('name') == joint_name:
                    # print(f"\tjoint body name >{joint.attrib.get('name')}<")
                    for origin in joint.findall('origin'):
                        # print(f"\tFound origin name >{origin.text}<")
                        return origin.text
                    
                    
def get_sdf_joint_lower_limit_from_a_sdf_model(model_name, joint_name, sdf_str):
    lower_limit = -0.0
    
    for model in sdf_str.findall('model'):
        if model.attrib.get('name') == model_name:
            # print(f"Found model >{model.attrib}<")
            for joint in model.findall('joint'):
                if joint.attrib.get('name') == joint_name:
                    # print(f"\tjoint body name >{joint.attrib.get('name')}<")
                    for axis in joint.findall('axis'):
                        # print(f"\tFound axis name >{axis.tag}<")
                        for limit in axis.findall('limit'):
                            # print(f"\tFound limit name >{limit.tag}<")
                            for lower in limit.findall('lower'):
                                # print(f"\tFound lower name >{lower.text}<")
                                lower_limit = lower.text
                                
    return lower_limit
                    
         
def get_sdf_joint_upper_limit_from_a_sdf_model(model_name, joint_name, sdf_str):
    upper_limit = 0.0
    
    for model in sdf_str.findall('model'):
        if model.attrib.get('name') == model_name:
            # print(f"Found model >{model.attrib}<")
            for joint in model.findall('joint'):
                if joint.attrib.get('name') == joint_name:
                    # print(f"\tjoint body name >{joint.attrib.get('name')}<")
                    for axis in joint.findall('axis'):
                        # print(f"\tFound axis name >{axis.tag}<")
                        for limit in axis.findall('limit'):
                            # print(f"\tFound limit name >{limit.tag}<")
                            for upper in limit.findall('upper'):
                                # print(f"\tFound upper name >{upper.text}<")
                                upper_limit = upper.text
                                
    return upper_limit        
         
         
                       
    
class SdfConverter(object):
    """Converts a sdf model to a Brax config."""

    def __init__(self, xml_string: AnyStr, add_collision_pairs: bool = False):
        ghum_xml = ET.fromstring(xml_string)
        self.body_tree = {}
        self.links = {}
        self.joints = {}
        self.config = config_pb2.Config()
        # self.brax_config = brax.Config(dt=0.05, substeps=100)
        #self.brax_config = config_pb2.Config()
        
        
        sdf_str = ghum_xml
        
        
        
        
        
        def create_bodies(sdf_str):
        
            model_names = get_all_model_names(sdf_str)
            # print(f"models >{model_names}<")
            
            
            
            for model_name in model_names:
                link_names = get_sdf_link_names_from_a_sdf_model(model_name, sdf_str)
            # print(f"body_names of >{model_names[0]}< are >{link_names}<")
            
            i = 0
            for link_name in link_names:
                geometry = get_sdf_geometry_from_a_sdf_model_link(model_names[0], link_name, sdf_str)
                # print(f"geometry of >{model_names[0]}< and >{link_name}< is >{geometry}<")
                
                size = get_sdf_box_size_from_a_sdf_model_link(model_names[0], link_name, geometry, sdf_str)
                # print(f"{geometry} size >{size}<")
                
                mass = get_sdf_mass_from_a_sdf_model_link(model_names[0], link_name, geometry, sdf_str)
                
                pose = get_sdf_pose_from_a_sdf_model_link(model_names[0], link_name, geometry, sdf_str)
            
                self.config.bodies.add(name=link_name)
                
                if geometry == 'sphere':
                    pass
                elif geometry == 'cylinder':
                    pass
                elif geometry == 'box':
                    self.config.bodies[i].inertia.x = 1
                    self.config.bodies[i].inertia.y = 1
                    self.config.bodies[i].inertia.z = 1
                    
                    self.config.bodies[i].mass = float(mass)
                    
                    c = self.config.bodies[i].colliders.add().box
                    c.halfsize.x = 1
                    c.halfsize.y = 1
                    c.halfsize.z = 1
                    
                i += 1
            
    
        def create_joints(sdf_str):
            model_names = get_all_model_names(sdf_str)
            for model_name in model_names:
                joint_names = get_sdf_joint_names_from_a_sdf_model(model_name, sdf_str)
                print(f"joint_names >{joint_names}<")
                
                i = 0
                for joint_name in joint_names:
                    self.config.joints.add(name=joint_name)
                    
                    joint_parent = get_sdf_joint_parent_from_a_sdf_model(model_name, joint_name, sdf_str)
                    self.config.joints[i].parent = joint_parent
                    print(f"joint_parent >{joint_parent}<")
                    
                # for joint_name in joint_names:
                    joint_child = get_sdf_joint_child_from_a_sdf_model(model_name, joint_name, sdf_str)
                    self.config.joints[i].child = joint_child
                    print(f"joint_child >{joint_child}<")
                    
                    joint_origin = get_sdf_joint_origin_from_a_sdf_model(model_name, joint_name, sdf_str)
                    # self.config.joints[i].child = joint_child
                    print(f"joint_origin >{joint_origin}<")
                    
                    lower_limit = get_sdf_joint_lower_limit_from_a_sdf_model(model_name, joint_name, sdf_str)
                    #del self.config.joints[i].angle_limit[:]
                    #self.config.joints[i].angle_limit.add(min=float(lower_limit), max=float(upper_limit))
                    print(f"lower_limit >{lower_limit}<")
                    
                    upper_limit = get_sdf_joint_upper_limit_from_a_sdf_model(model_name, joint_name, sdf_str)
                    #self.config.joints[i].angle_limit.add(max=float(upper_limit))
                    self.config.joints[i].angle_limit.add(min=float(lower_limit), max=float(upper_limit))
                    print(f"upper_limit >{upper_limit}<")
                    
                    i += 1
            
        create_bodies(sdf_str)
        
        create_joints(sdf_str)
        
        
        
        
        
        
        