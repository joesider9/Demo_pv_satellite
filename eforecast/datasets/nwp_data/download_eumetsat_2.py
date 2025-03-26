import os
import eumdac
import requests
import time
import fnmatch
import shutil
import pandas as pd
from joblib import Parallel
from joblib import delayed
import astral
from astral.sun import sun

# Each product is a tuple (long name, short name and a dict with bands
class DownloadEUMETSAT:
    def __init__(self, date, path_sat, roi=None, products=None):
        self.start = date - pd.DateOffset(hours=1) + pd.DateOffset(minutes=5)
        self.end = date + pd.DateOffset(minutes=5)
        self.date = date
        self.path_sat = path_sat
        self.perform = True
        if isinstance(roi, list):
            roi = {"NSWE" : roi}
        self.roi = roi

        l = astral.LocationInfo(name='Athens', region='Athens', timezone='Europe/Athens',
                                latitude=38, longitude=23.8)
        sun_attr = sun(l.observer, date=date)

        sunrise = pd.to_datetime(sun_attr['dawn'].strftime('%Y%m%d %H:%M'), format='%Y%m%d %H:%M')
        sunset = pd.to_datetime(sun_attr['dusk'].strftime('%Y%m%d %H:%M'), format='%Y%m%d %H:%M')
        if sunrise <= date <= sunset:
            self.perform = True
        else:
            self.perform = False

        hseveri = [
            ('EO:EUM:DAT:MSG:HRSEVIRI',
             'HRSEVIRI',
             {"bands": [f"channel_{i}" for i in [1, 2, 3, 4, 7, 8, 9, 10 ,11]]}, 'IR')]

        if products is None:
            self.products = hseveri
        else:
            self.products = products
        self.path_file = os.path.join(path_sat, f'{date.year}_{date.strftime("%B")}_{date.day}', f'{date.hour}')
        if not os.path.exists(self.path_file):
            os.makedirs(self.path_file)
        self.refresh_token()



    def refresh_token(self):
        consumer_key = 'GqmOLQZkOY0NWG7WoKVJFqkqxM8a'
        consumer_secret = 'KH6OoF46Rts_56UyfNSiBfjqHFEa'

        credentials = (consumer_key, consumer_secret)

        self.token = eumdac.AccessToken(credentials)

    def status(self, customisation):
        try:
            status = customisation.status
        except:
            self.remove_customizations()
            return 'wrong'
        sleep_time = 10  # seconds

        # Customisation Loop
        while status:
            # Get the status of the ongoing customisation
            try:
                status = customisation.status
            except:
                self.remove_customizations()
                return 'wrong'
            if "DONE" in status:
                print(f"Customisation {customisation._id} is successfully completed.")
                break
            elif status in ["ERROR", "FAILED", "DELETED", "KILLED", "INACTIVE"]:
                print(f"Customisation {customisation._id} was unsuccessful. Customisation log is printed.\n")
                print(customisation.logfile)
                break
            elif "QUEUED" in status:
                print(f"Customisation {customisation._id} is queued.")
            elif "RUNNING" in status:
                print(f"Customisation {customisation._id} is running.")
            time.sleep(sleep_time)
        return 'Done'

    def remove_customizations(self):
        self.refresh_token()
        datatailor = eumdac.DataTailor(self.token)
        for customisation in datatailor.customisations:
            if customisation.status in ['QUEUED', 'INACTIVE', 'RUNNING']:
                customisation.kill()
                print(
                    f'Delete {customisation.status} customisation {customisation} from {customisation.creation_time} UTC.')
                try:
                    customisation.delete()
                except Exception as error:
                    print("Unexpected error:", error)
            else:
                print(f'Delete completed customisation {customisation} from {customisation.creation_time} UTC.')
                try:
                    customisation.delete()
                except requests.exceptions.RequestException as error:
                    print("Unexpected error:", error)

    def download(self):
        if not self.perform:
            return
        print(f'Downloading {self.end} UTC')
        for product in self.products:
            if len(os.listdir(self.path_file)) >= 4:
                return
            datastore = eumdac.DataStore(self.token)
            try:
                selected_collection = datastore.get_collection(product[0])
                print(f"{selected_collection} - {selected_collection.title}")
            except:
                print('Cannot find collection')
                continue
            items = selected_collection.search(dtstart=self.start, dtend=self.end)
            datatailor = eumdac.DataTailor(self.token)
            if len(product[2]) > 0:
                chain = eumdac.tailor_models.Chain(product=product[1],
                                                   filter=product[2],
                                                   format='hdf5',
                                                   roi=self.roi
                                                   )
            else:
                chain = eumdac.tailor_models.Chain(product=product[1],
                                                   format='hdf5',
                                                   roi=self.roi
                                                   )
            for i, item in enumerate(items):
                # if i % 2 == 1:
                #     continue
                try:
                    print(item)
                except:
                    continue
            for i, item in enumerate(items):
                # if i % 2 == 1:
                #     continue
                status = 'Undone'
                trials = 0
                while status != 'Done':
                    if trials < 10:
                        customisation = datatailor.new_customisation(item, chain)
                        status = self.status(customisation)
                        trials += 1
                    else:
                        break
                if status != 'Done':
                    continue
                try:
                    tiff, = fnmatch.filter(customisation.outputs, '*.h5')
                    jobID = customisation._id

                    with customisation.stream_output(tiff,) as stream:
                        stream_name = stream.name.split('_')
                        if len(product[2]) == 0:
                            filename = 'Cloud' + '_' + '_'.join(stream_name[1:3]) + '.h5'
                        else:
                            band = product[3]
                            filename = 'HRSEVERI' + '_' + band + '_' +  '_'.join(stream_name[1:3]) + '.h5'
                        with open(os.path.join(self.path_file, filename), mode='wb') as fdst:
                            print(f"Downloading the H5 output of the customisation {stream.name} for date {self.end}")
                            shutil.copyfileobj(stream, fdst)
                    print(f"Downloaded the H5 output of the customisation {jobID} for date {self.end}")
                    customisation.delete()
                except:
                    self.remove_customizations()
                    continue

def download(date, path_sat, roi_greece):
    try:
        downloader = DownloadEUMETSAT(date, path_sat, roi= roi_greece)
        downloader.download()
    except Exception as e:
        print(e)
        pass


def run_sat_download(dates, static_data):
    path_sat = static_data['sat_folder']
    if not os.path.exists(path_sat):
        os.makedirs(path_sat)
    downloader = DownloadEUMETSAT(dates[0], path_sat, roi=static_data['image_coord'])
    downloader.remove_customizations()
    for date in dates:
        print(date)
        download(date, path_sat, static_data['image_coord'])

    # Parallel(n_jobs=3)(delayed(download)(date, path_sat, static_data['image_coord']) for date in dates)

    downloader.remove_customizations()
