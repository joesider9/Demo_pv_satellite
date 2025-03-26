import os
import ftplib
import datetime, time
import pandas as pd
import pygrib, h5py, math
import numpy as np
from pytz import timezone


def datetime_exists_in_tz(dt, tz):
    try:
        dt.tz_localize(tz)
        return True
    except:
        return False


def skiron_download(now=None, path_nwp=None):
    if path_nwp is None:
        path_nwp = '/nwp'
    if now is None:
        now = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y'), format='%d%m%y')

    with ftplib.FTP('ftp.mg.uoa.gr') as ftp:
        try:
            ftp.login('mfstep', '!lam')
            ftp.set_pasv(True)

        except:
            print('Error in connection to FTP')
        local_dir = path_nwp + '/' + now.strftime('%Y') + '/' + now.strftime('%d%m%y')
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        for hor in range(76):
            try:
                target_filename = '/forecasts/Skiron/daily/005X005/' + now.strftime(
                    '%d%m%y') + '/MFSTEP005_00' + now.strftime('%d%m%y') + '_' + str(hor).zfill(3) + '.grb'

                local_filename = local_dir + '/MFSTEP005_00' + now.strftime('%d%m%y') + '_' + str(hor).zfill(3) + '.grb'
                if not os.path.exists(local_filename):
                    with open(local_filename, 'w+b') as f:
                        res = ftp.retrbinary('RETR %s' % target_filename, f.write)
                        count = 0
                        while not res.startswith('226 Transfer complete') and count <= 4:
                            time.sleep(60)
                            print('Downloaded of file {0} is not compile.'.format(target_filename))
                            os.remove(local_filename)
                            with open(local_filename, 'w+b') as f:
                                res = ftp.retrbinary('RETR %s' % target_filename, f.write)
                            count += 1
            except:
                print('Error downloading  {0} '.format(local_filename))
                continue
        ftp.quit()
