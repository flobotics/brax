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
        print(f"models >{model_names}<")
        
        
        
        for model_name in model_names:
            link_names = get_sdf_link_names_from_a_sdf_model(model_name, sdf_str)
        print(f"body_names of >{model_names[0]}< are >{link_names}<")
        
        i = 0
        for link_name in link_names:
            geometry = get_sdf_geometry_from_a_sdf_model_link(model_names[0], link_name, sdf_str)
            print(f"geometry of >{model_names[0]}< and >{link_name}< is >{geometry}<")
            
            size = get_sdf_box_size_from_a_sdf_model_link(model_names[0], link_name, geometry, sdf_str)
            print(f"{geometry} size >{size}<")
        
            self.config.bodies.add(name=link_name)
            
            if geometry == 'sphere':
                self.config.bodies[link_name].inertia.x = 1
            elif geometry == 'cylinder':
                self.config.bodies[link_name].inertia.x = 1
            elif geometry == 'box':
                self.config.bodies[i].inertia.x = 1
                
            i += 1
        
        # geometry = get_sdf_geometry_from_a_sdf_model_link(model_names[0], link_names[1], sdf_str)
        # print(f"geometry of >{model_names[0]}< and >{link_names[1]}< is >{geometry}<")
        
        
        # print(f"geometry >{geometry}<")
        # size = get_sdf_box_size_from_a_sdf_model_link(model_names[0], link_names[0], geometry, sdf_str)
        # print(f"{geometry} size >{size}<")
        
    create_bodies(sdf_str)
    
    
    