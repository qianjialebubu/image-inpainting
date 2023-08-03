import oss2
import os
import datetime


class OSS(object):
    """定义一个简单的oss操作类，支持文件上传和下载"""

    def __init__(self, accessKey_id, accessKey_secret, endpoint, bucket_name):
        self.auth = oss2.Auth(accessKey_id, accessKey_secret)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket_name)

    def download_from_oss(self, oss_folder_prefix, object_name, local_save_path):
        """拼接本地保存时的文件路径，且保持oss中指定目录以下的路径层级"""
        oss_path_prefix = object_name.split(oss_folder_prefix)[-1]  # oss原始路径,以'/'为路径分隔符
        oss_path_prefix = oss_path_prefix.split('?')[0]
        oss_path_prefix = os.sep.join(oss_path_prefix.strip('/').split('/'))  # 适配win平台
        local_file_path = os.path.join(local_save_path, oss_path_prefix)
        local_file_prefix = local_file_path[:local_file_path.rindex(os.sep)]  # 本地保存文件的前置路径，如果不存在需创建
        if not os.path.exists(local_file_prefix):
            os.makedirs(local_file_prefix)

        self.bucket.get_object_to_file(object_name, local_file_path)

    def upload_to_oss(self, prefix, suffix, local_upload_path):
        """上传指定路径下的目录或文件，如果oss路径不存在，则自动创建"""
        # 当前日期时间作为最新上传的目录名
        folder_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        oss_upload_prefix = prefix.rstrip('/') + '/' + folder_name
        # 遍历指定上传目录文件，并上传
        for root, dirs, files in os.walk(local_upload_path):
            local_upload_path = local_upload_path.rstrip(os.sep)  # 去除外部输入时结尾可能带入的路径符号
            for file in files:
                file_path = os.path.join(root, file)
                relative_file_path = file_path.split(local_upload_path)[1]  # 保持upload目录下的路径层级
                relative_file_path = relative_file_path.strip(os.sep)
                oss_relative_path = relative_file_path.replace(os.sep, '/')  # 转换成oss的路径格式，适配linux\win
                oss_upload_path = oss_upload_prefix + '/' + oss_relative_path
                # 上传该文件
                if file.endswith(suffix):
                    self.bucket.put_object_from_file(oss_upload_path, file_path)

    def travel_download(self, prefix, suffix, local_save_path):
        """
        :param prefix: oss目录前缀，即遍历以prefix开头的文件
        :param suffix: 文件后缀名，如，.csv，指定下载何种类型的文件
        :param local_save_path: 下载文件的保存路径
        :return:
        """
        # 下载指定目录下的指定后缀的文件，且保存时维持目录层级格式
        # 列举指定prefix目录下的层级目录，定位到目标目录后，再做深度遍历
        local_save_path = local_save_path.rstrip(os.sep)  # 去除外部输入时结尾可能带入的路径符号
        top_level_folder = []
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix, delimiter='/'):
            if obj.is_prefix():
                # 目录
                top_level_folder.append(obj.key)
            else:
                # 文件
                pass

        # 获取最近一次更新的目录,并下载该目录及其子目录下指定后缀的文件
        target_folder = max(top_level_folder)

        for obj in oss2.ObjectIterator(self.bucket, prefix=target_folder):
            if obj.is_prefix():
                # 目录
                continue
            else:
                # 只下载指定后缀的文件，oss中xxx/xxx/也会被认为是文件，根据prefix而定
                if obj.key.endswith(suffix):
                    # 下载
                    self.download_from_oss(target_folder, obj.key, local_save_path)
    def download_from_oss2(self, object_name, local_file_path):
        self.bucket.get_object_to_file(object_name, local_file_path)
