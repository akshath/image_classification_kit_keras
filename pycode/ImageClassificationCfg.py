import yaml
import pycode.FileIOUtil

class ImageClassificationCfg:
    def __init__(self, file):
        self.cfg_file = file

    def read(self, cfg_file=None):
        #read cfg
        if cfg_file is None:
            cfg_file = self.cfg_file
        with open(cfg_file, "r") as ymlfile:
            try:
                self.cfg = yaml.load(ymlfile, Loader=yaml.CLoader)
            except AttributeError:
                self.cfg = yaml.load(ymlfile)

    def load(self, cfg_file=None):
        self.read(cfg_file)

        project_name = self.cfg["project_name"]
        project_parent_dir = self.cfg["project_parent_dir"]

        if 'project_dir' in self.cfg.keys():
            self.project_dir = self.cfg["project_dir"]
        else:
            self.project_dir = project_parent_dir + project_name + "/"
            self.cfg["project_dir"] = self.project_dir 

        if 'project_data_dir' in self.cfg.keys():
            self.project_data_dir = self.cfg["project_data_dir"]
        else:
            self.project_data_dir = self.project_dir + "data/"
            self.cfg["project_data_dir"] = self.project_data_dir
            
        self.project_temp_dir = self.cfg["temp_dir"] + project_name + "/"
        self.loc_unknown = self.project_temp_dir+'non-labeled/'

        labels_from_dir = self.cfg['labels_from_dir']
        if labels_from_dir==True:
            self.labels = pycode.FileIOUtil.get_dir(self.project_data_dir, only_dir=True)
        else:
            self.labels = self.cfg['labels'].split(' ')
        self.labels.sort()
        self.cfg['labels'] = self.labels

        model_file = self.cfg['model_file']
        if model_file is not None:
            if model_file.startswith('./'):
                model_file = self.project_dir + model_file[2:]
                self.cfg['model_file'] = model_file

    def log_info(self):
        print('project_name: ',self.cfg["project_name"])
        print('-'*20)

        print('project_parent_dir: ',self.cfg["project_parent_dir"])
        print('project_data_dir: ',self.cfg["project_data_dir"])
        print('temp_dir: ', self.cfg["temp_dir"])
        print('file_ext: ', self.cfg["file_ext"])
        print('-'*20)

        print('reduce_image_wh_by: ',self.cfg['reduce_image_wh_by'])
        print('crop_image_from_left: ',self.cfg['crop_image_from_left'])
        print('crop_image_from_right: ',self.cfg['crop_image_from_right'])
        print('-'*20)

        print('project_dir: ', self.project_dir)
        print('project_temp_dir:', self.project_temp_dir)
        print('loc_unknown:', self.loc_unknown)
        print('-'*20)

        print('labels_from_dir: ',self.cfg['labels_from_dir'])
        print('label count: ', len(self.labels))
        print('labels:', self.labels)

if __name__ == '__main__':
    print('Testing..')
    cfg = ImageClassificationCfg('./project/home_presence.yml')
    cfg.load()
    cfg.log_info()
