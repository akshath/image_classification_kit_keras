import yaml
import pycode.FileIOUtil

class ImageClassificationCfg:
    def __init__(self, file):
        self.cfg_file = file

    def load(self, cfg_file=None):
        #read cfg
        if cfg_file is None:
            cfg_file = self.cfg_file
        with open(cfg_file, "r") as ymlfile:
            try:
                self.cfg = yaml.load(ymlfile, Loader=yaml.CLoader)
            except AttributeError:
                self.cfg = yaml.load(ymlfile)

        project_name = self.cfg["project_name"]
        project_parent_dir = self.cfg["project_parent_dir"]

        self.project_dir = project_parent_dir + project_name + "/"
        if self.project_data_dir is None:
            self.project_data_dir = self.project_dir + "data/"
        self.project_temp_dir = self.cfg["temp_dir"] + project_name + "/"
        self.loc_unknown = self.project_temp_dir+'non-labeled/'

        labels_from_dir = self.cfg['labels_from_dir']
        if labels_from_dir==True:
            self.labels = pycode.FileIOUtil.get_dir(self.project_data_dir, only_dir=True)
        else:
            self.labels = self.cfg['labels'].split(' ')
        self.labels.sort()
        self.cfg['labels'] = self.labels

    def log_info(self):
        print('project_name: ',self.cfg["project_name"])
        print('-'*20)

        print('project_parent_dir: ',self.cfg["project_parent_dir"])
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
