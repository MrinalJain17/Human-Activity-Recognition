import urllib
import os
from keras.utils.data_utils import _extract_archive
from tqdm import tqdm


class TqdmUpTo(tqdm):

    """
    Code of this class taken from - https://pypi.python.org/pypi/tqdm
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Parameters:
            b (int): optional
                Number of blocks transferred so far [default: 1]

            bsize (int): optional
                Size of each block (in tqdm units) [default: 1]

            tsize (int): optional
                Total size (in tqdm units)
                If [default: None] remains unchanged
        """

        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_files():
    """
    Downloads the 6 zip files in the current directory and extracts them within a directory called 'Data'

    The format mentioned in the file `Directory Structure for Data.txt` is maintained.

    If a file is already present, it is not downloaded again.
    """

    base_link = r'http://www.nada.kth.se/cvap/actions/'
    base_file = os.getcwd() + r'/'
    files = ['walking', 'jogging', 'running',
             'boxing', 'handwaving', 'handclapping']

    for file in files:

        link = base_link + file + '.zip'
        file_name = base_file + file + '.zip'

        if not os.path.exists(file_name):

            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=link.split('/')[-1]) as t:
                urllib.request.urlretrieve(
                    link, file_name, reporthook=t.update_to, data=None)

        success = _extract_archive(file_name, path=r'Data/' + file)

        if success:
            print('-----------------------------{}------'.format('-' * len(file)))
            print('| Successfully extracted --> {}.zip |'.format(file))
            print('-----------------------------{}------'.format('-' * len(file)))
            os.remove(file_name)
        else:
            print('\nUnsuccessful extraction --> {}.zip\n'.format(file))
