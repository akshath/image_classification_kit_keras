import yaml
import FileIOUtil

class ImageClassificationCfg:
    def __init__(self, file):
        self.cfg_file = file
    
    def load(self, cfg_file=None):
        #read cfg
        if cfg_file is None:
            cfg_file = self.cfg_file
        with open(cfg_file, "r") as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.CLoader)
            
        labels_from_dir = self.cfg['labels_from_dir']
        if labels_from_dir==True:
            self.labels = FileIOUtil.get_dir(project_dir, only_dir=True)
        else:
            self.labels = self.cfg['labels'].split(' ')
        self.labels.sort()
        self.cfg['labels'] = self.labels
            
    def log_info(self):
        project_name = self.cfg["project_name"]
        print('project_name: ',project_name)
        print('-'*20)

        project_parent_dir = self.cfg["project_parent_dir"]
        
        print('project_parent_dir: ',project_parent_dir)
        print('temp_dir: ', self.cfg["temp_dir"])
        print('file_ext: ', self.cfg["file_ext"])
        print('-'*20)

        reduce_image_wh_by = self.cfg['reduce_image_wh_by']
        print('reduce_image_wh_by: ',reduce_image_wh_by)
        crop_image_from_left = self.cfg['crop_image_from_left']
        crop_image_from_right = self.cfg['crop_image_from_right']
        print('crop_image_from_left: ',crop_image_from_left)
        print('crop_image_from_right: ',crop_image_from_right)
        print('-'*20)

        self.project_dir = project_parent_dir + project_name + "/"
        self.project_temp_dir = self.cfg["temp_dir"] + project_name + "/"
        self.loc_unknown = self.project_temp_dir+'non-labeled/'
        print('project_dir: ', self.project_dir)
        print('project_temp_dir:', self.project_temp_dir)
        print('loc_unknown:', self.loc_unknown)
        print('-'*20)

        labels_from_dir = self.cfg['labels_from_dir']
        print('labels_from_dir: ',labels_from_dir)
        print('label count: ', len(self.labels))
        print('labels:', self.labels)

if __name__ == '__main__':
    print('Testing..')
    cfg = ImageClassificationCfg('./project/home_presence.yml')
    cfg.load()
    cfg.log_info()