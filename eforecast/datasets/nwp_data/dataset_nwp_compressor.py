from eforecast.datasets.nwp_data.nwp_compressors.load_compressor import LoadCompressor
from eforecast.datasets.nwp_data.nwp_compressors.minimal_compressor import MinimalCompressor
from eforecast.datasets.nwp_data.nwp_compressors.dense_compressor import DenseCompressor
from eforecast.datasets.nwp_data.nwp_compressors.semi_full_compressor import SemiFullCompressor
from eforecast.datasets.nwp_data.nwp_compressors.full_compressor import FullCompressor


class DatasetNWPsCompressor:
    def __init__(self, static_data, nwp_data, nwp_metadata, compress):
        self.static_data = static_data
        self.nwp_data = nwp_data
        self.nwp_metadata = nwp_metadata
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.path_data = self.static_data['path_data']
        self.horizon = self.static_data['horizon']
        self.horizon_type = self.static_data['horizon_type']
        self.use_data_before_and_after_target = self.static_data['use_data_before_and_after_target']
        self.nwp_models = self.static_data['NWP']
        self.area_group = self.static_data['area_group']
        self.areas = self.static_data['NWP'][0]['area']

        self.type = self.static_data['type']
        self.compress_method = compress

        self.variables = dict([(var_data['name'], var_data) for var_data in static_data['variables']
                               if var_data['type'] == 'nwp'])

    def compress(self):
        if self.compress_method == 'load':
            compressor = LoadCompressor(self.static_data, self.nwp_data, self.nwp_metadata)
            nwp_compressed_all, nwp_compressed = compressor.load_compress()
        elif self.compress_method == 'minimal':
            compressor = MinimalCompressor(self.static_data, self.nwp_data, self.nwp_metadata)
            nwp_compressed_all, nwp_compressed = compressor.dense_compress()
        elif self.compress_method == 'dense':
            compressor = DenseCompressor(self.static_data, self.nwp_data, self.nwp_metadata)
            nwp_compressed_all, nwp_compressed = compressor.dense_compress()
        elif self.compress_method == 'semi_full':
            compressor = SemiFullCompressor(self.static_data, self.nwp_data, self.nwp_metadata)
            nwp_compressed_all, nwp_compressed = compressor.semi_full_compress()
        elif self.compress_method == 'full':
            compressor = FullCompressor(self.static_data, self.nwp_data, self.nwp_metadata)
            nwp_compressed_all, nwp_compressed = compressor.full_compress()
        else:
            raise NotImplementedError(f'Compress method {self.compress_method} not implemented yet')
        return nwp_compressed_all, nwp_compressed
