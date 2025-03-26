import pvlib

import numpy as np
import pandas as pd
import cv2

from pvlib.location import Location


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped


def get_lats_longs_by_date(nwps, p_dates, area_group, areas, nwp_resolution):
    nwp = None
    flag = False
    for date in p_dates:
        if date in nwps:
            nwp = nwps[date]
            flag = True
            break
    if not flag:
        lats = None
        longs = None
        return lats, longs
    if nwp is None:
        raise ValueError('Cannot find nwps in nwp dictionary in order to find lats longs')

    lat, long = nwp['lat'], nwp['long']
    if len(lat.shape) == 1:
        lat = lat.reshape(-1, 1)
        long = long.reshape(1, -1)
    if isinstance(areas, list):
        # Is this guaranteed to be 5x5 ? I think yes, because of the resolution. TODO: VERIFY
        lats = np.where((lat[:, 0] >= areas[0][0]) & (lat[:, 0] <= areas[1][0]))[0]
        longs = np.where((long[0, :] >= areas[0][1]) & (long[0, :] <= areas[1][1]))[0]
    else:
        lats = dict()
        longs = dict()
        for area in sorted(areas.keys()):
            lats[area] = \
                np.where((lat[:, 0] >= areas[area][0][0]) & (lat[:, 0] <= areas[area][1][0]))[0]
            longs[area] = \
                np.where((long[0, :] >= areas[area][0][1]) & (long[0, :] <= areas[area][1][1]))[0]
        lats['area_group'] = \
            np.where((lat[:, 0] >= area_group[0][0]) & (
                    lat[:, 0] <= area_group[1][0] + nwp_resolution / 2))[0]
        longs['area_group'] = \
            np.where((long[0, :] >= area_group[0][1]) & (
                    long[0, :] <= area_group[1][1] + nwp_resolution / 2))[0]
    return lats, longs


def compute_area_grid(lat, long, resolution, round_coord, levels):
    lat_range = np.arange(np.around(lat, round_coord) - 20, np.around(lat, round_coord) + 20,
                          resolution)
    lat1 = lat_range[np.abs(lat_range - lat).argmin()] - resolution / 4
    lat2 = lat_range[np.abs(lat_range - lat).argmin()] + resolution / 4

    long_range = np.arange(np.around(long, round_coord) - 20, np.around(long, round_coord) + 20,
                           resolution)
    long1 = long_range[np.abs(long_range - long).argmin()] - resolution / 4
    long2 = long_range[np.abs(long_range - long).argmin()] + resolution / 4

    return [[lat1 - resolution * levels, long1 - resolution * levels],
            [lat2 + resolution * levels, long2 + resolution * levels]]


def create_area(coord, nwp_resolution):
    levels = 4 if nwp_resolution == 0.05 else 2
    round_coord = 1 if nwp_resolution == 0.05 else 0

    coord_temp = []
    if coord is None:
        area = dict()
    elif isinstance(coord, list):
        if len(coord) == 2:
            lat, long = coord[0], coord[1]
            area = compute_area_grid(lat, long, nwp_resolution, round_coord, levels)
            coord_temp = coord
        elif len(coord) == 4:
            area = list(np.array(coord).reshape(2, 2))
            coord_temp = np.mean(np.array(coord).reshape(2, 2), axis=0).tolist()
        else:
            raise ValueError(
                'Wrong coordinates. Should be point (lat, long) or area [lat1, long1, lat2, long2]')
    elif isinstance(coord, dict):
        area = dict()
        coord_temp = [0, 0]
        for key, value in coord.items():
            if len(value) == 4:
                area[key] = np.array(value).reshape(2, 2).tolist()
                value1 = np.mean(np.array(value).reshape(2, 2), axis=0).tolist()
                coord_temp = [coord_temp[0] + value1[0], coord_temp[1] + value1[1]]
            else:
                raise ValueError(
                    'Wrong coordinates. Should be area [lat1, long1, lat2, long2]')
        coord_temp = [coord_temp[0] / len(coord), coord_temp[1] / len(coord)]
    else:
        raise ValueError('Wrong coordinates. Should be dict or list')
    print('Areas created successfully')

    return area, coord_temp


def check_empty_nwp(nwp, variables):
    flag = True
    for var in variables:
        if nwp[var].shape[0] == 0:
            print(var)
            flag = False
            break
    return flag


def check_empty_multiple_nwp(nwp, variables):
    flag = True
    for var in variables.keys():
        if nwp[var].shape[0] == 0:
            flag = False
            break
    return flag


def clear_sky(date, lat, long, local_time_zone, site_time_zone):
    dates = pd.date_range(start=date - pd.DateOffset(hours=24), end=date + pd.DateOffset(hours=24),
                          freq='h', tz=site_time_zone)
    loc = Location(lat, long, site_time_zone, 0, 'kjhkjhk')

    ck1 = loc.get_clearsky(dates)
    ghi_ = ck1.ghi.tz_convert(local_time_zone)
    d1 = [pd.to_datetime(dt.strftime('%d%m%y%H%M'), format='%d%m%y%H%M') for dt in ghi_.index]
    ghi_.index = d1
    return ghi_.max()


def get_clear_sky(dates, lat, long, local_time_zone, site_time_zone, ts_resolution):
    dates_peak = dates.round('D').unique()
    if dates_peak.shape[0] == 1:
        dates_peak = pd.DatetimeIndex([dates_peak[0] - pd.DateOffset(days=1), dates_peak[0] + pd.DateOffset(days=2)])
    cs = np.array([clear_sky(d, lat, long, local_time_zone, site_time_zone) for d in dates_peak])
    cs = pd.DataFrame(cs, index=dates_peak, columns=['clear_sky'])
    dates_new = pd.date_range(min(dates[0], dates_peak[0]), max(dates[-1], dates_peak[-1]), freq=ts_resolution)
    cs_new = pd.DataFrame(index=dates_new, columns=['clear_sky'])
    cs_new.loc[cs.index] = cs
    cs_new = cs_new.astype('float')
    cs_new = cs_new.interpolate(method='nearest', limit_direction='both')
    return cs_new


def bcs(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    alpha = 1.2
    beta = 1.5
    trans = cv2.filter2D(cv2.convertScaleAbs(img, alpha, beta), -1, kernel)
    max_img = trans.max()
    loops = 0
    while max_img > 255 or max_img < 220:
        if max_img < 220:
            alpha = alpha + 0.05
            beta = beta + 0.1
        elif max_img > 250:
            alpha = alpha - 0.1
            beta = beta - 0.2
        trans = cv2.filter2D(cv2.convertScaleAbs(img, alpha, beta), -1, kernel)
        max_img = trans.max()
        loops += 1
        if loops > 100:
            print('Image errors')
            return None
    return trans

def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    data_range = np.arange(len(data))
    idx_list = data_range[s >= m]
    return data[s < m], idx_list