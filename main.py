import datetime
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import scipy.signal as signal
import shapefile
import soundfile as sf
from PIL import Image, ImageDraw
from pyorbital.orbital import Orbital
from scipy import interpolate


def rate_grafic(_data):
    x = []
    data_temp = _data[len(_data) // 2:len(_data) // 2 + fs]
    for i in range(fs):
        x.append(i)
    temp_x = np.arange(0, len(data_temp))
    plt.plot(temp_x, data_temp)
    plt.savefig("rate_grafic.png")


# smoothing rate peak
def hilbert(_data):
    analytical_signal = signal.hilbert(_data)
    amplitude_envelope = np.abs(analytical_signal)
    return amplitude_envelope


def full_image(_smooth_data, title):
    matrix = []
    begin = 0
    end = len(_smooth_data) - fs // 2
    for i in range(begin, end, fs // 2):
        data_is = _smooth_data[i:i + fs // 2]
        matrix.append(data_is)
    matrix = np.array(matrix)
    pil_image = Image.fromarray(matrix * 255).convert('RGB')
    pil_image.save('Pil_image' + title + '.png')


def straighter(_smooth_data):
    move = 1941
    counter = 0
    matrix = []
    begin = len(_smooth_data) // 8
    end = len(_smooth_data) - fs // 2
    for i in range(begin, end, fs // 2):
        data_temp = _smooth_data[i: i + fs // 2]
        counter += 0.5
        temp_count = np.round(counter - 0.1, 0)
        data_begin = data_temp[move + int(temp_count):-1]
        data_end = data_temp[:move + int(temp_count) - 1]
        data_begin = np.array(data_begin)
        data_end = np.array(data_end)
        data_end = data_end.tolist()
        data_begin = data_begin.tolist()
        data_begin += data_end
        matrix.append(data_begin)
    matrix = np.array(matrix)
    return matrix


def image_area(_smooth_data, _cut):
    move = 1640
    num = 105
    counter = 0
    matrix = []
    begin = len(_smooth_data) // _cut
    _smooth_data = np.zeros(len(_smooth_data))
    end = len(_smooth_data) - len(_smooth_data) // _cut - fs // 2
    for i in range(begin + fs // 2 * 100, end, fs // 2):
        counter += 0.5
        temp_const = np.round(counter - 0.1, 0)
        data_is = _smooth_data[i + move + int(temp_const): i + move + num + int(temp_const)]

        matrix.append(data_is)
    matrix = np.array(matrix)
    Pil_Image = Image.fromarray(matrix * 255).convert('RGB')
    Pil_Image.show()


def normalize(matrix):
    matrix_telemetria = [0] * 8
    matrix = np.array(matrix)
    data_telemetria = []
    for i in range(16):
        matrix_telemetria[0] = matrix[284 + 8 * i][2634:2650]
        matrix_telemetria[1] = matrix[285 + 8 * i][2634:2650]
        matrix_telemetria[2] = matrix[286 + 8 * i][2634:2650]
        matrix_telemetria[3] = matrix[287 + 8 * i][2634:2650]
        matrix_telemetria[4] = matrix[288 + 8 * i][2634:2650]

        matrix_telemetria[5] = matrix[289 + 8 * i][2634:2650]
        matrix_telemetria[6] = matrix[290 + 8 * i][2634:2650]
        matrix_telemetria[7] = matrix[291 + 8 * i][2634:2650]
        data_telemetria.append(np.mean(matrix_telemetria))
    data_telemetria = np.array(data_telemetria)
    white = np.min(data_telemetria)
    black = np.max(data_telemetria)
    return np.abs(matrix - white) / np.abs(black - white)


def norm_straighter(_smooth_data):
    matrix = []
    move_file = open('smooth_points.txt', 'r')
    move_mass = []
    move_x_mass = []
    try:
        move = move_file.read().split('\n')
        move_mass = move[0].split(' ')
        move_x_mass = move[1].split(' ')
    except Exception:
        print("error with smooth_points.txt read")
    else:
        print("smooth_points.txt read success")
    finally:
        move_file.close()
    move_mass = np.array(move_mass, dtype=int)
    move_x_mass = np.array(move_x_mass, dtype=int)
    smooth_degree = 5
    move_mass = np.polyfit(move_x_mass, move_mass, smooth_degree)
    model_mass = []
    for element in range(0, 733, 1):
        model_mass.append(np.round(np.polyval(move_mass, element), 0))
    model_mass = np.array(model_mass, dtype=int)
    smooth_matrix = straighter(_smooth_data)
    for element in range(0, 733, 1):
        data_begin = []
        data_end = []
        counter = 0
        data_begin = smooth_matrix[element][model_mass[element]:]
        data_end = smooth_matrix[element][:model_mass[element] - 1]
        data_begin = np.array(data_begin)
        data_end = np.array(data_end)
        data_begin = data_begin.tolist()
        data_end = data_end.tolist()
        data_begin += data_end
        matrix.append(data_begin)
        counter += 0.5
    matrix = np.array(matrix)
    return matrix

def find_latlon():
    tle_file = open('TLE.txt', 'r')
    try:
        tle = tle_file.read()
        tle = tle.split("\n")
        orb = Orbital("sattelite", tle_file=None, line1=tle[0], line2=tle[1])
    except Exception:
        print("error with tle file")
    else:
        print("tle read")
    finally:
        tle_file.close()

    start = datetime.date(2022, 4, 18)

    time = datetime.datetime.combine(start, datetime.time(0, 0))
    time = time + datetime.timedelta(seconds=55803)
    lon_start, lat_start, alt_start = orb.get_lonlatalt(time)
    time = time + datetime.timedelta(seconds=419)
    lon_end, lat_end, alt_end = orb.get_lonlatalt(time)
    lon_start_r = lon_start * np.pi / 180
    lat_start_r = lat_start * np.pi / 180
    lon_end_r = lon_end * np.pi / 180
    lat_end_r = lat_end * np.pi / 180

    _latlon_r = np.array(([lat_start_r, lon_start_r], [lat_end_r, lon_end_r]), dtype=float)
    _lonlat_g = np.array(([lon_start, lat_start], [lon_end, lat_end]), dtype=float)

    return _latlon_r, _lonlat_g

def paint_image(matrix):
    maxx_in_line = []
    maxx_in_column = []
    minn_in_line = []
    minn_in_column = []
    for i in range(len(matrix)):
        maxx_in_line.append(np.max(matrix[i, 2989:5383]))
        minn_in_line.append(np.min(matrix[i, 2989:5383]))
    maxx_in_line = np.array(maxx_in_line)
    maxx_in_line_v = np.max(maxx_in_line)
    minn_in_line_v = np.min(minn_in_line)
    for i in range(2395):
        maxx_in_column.append(np.max(matrix[:, 228 + i]))
        minn_in_column.append(np.min(matrix[:, 228 + i]))
    maxx_in_column = np.array(maxx_in_column)
    maxx_in_column_v = np.max(maxx_in_column)
    minn_in_column_v = np.min(minn_in_column)
    maxx = maxx_in_line_v
    colorfull = Image.open('WXtoImg-NO.png')
    pix = colorfull.load()
    pil_image = Image.fromarray(matrix * 255).convert('RGB')
    draw = ImageDraw.Draw(pil_image)

    matrix_3d = np.arange(733 * len(matrix[0]) * 3)
    matrix_3d = matrix_3d.reshape((733, len(matrix[0]), 3))

    for i in range(733):
        for j in range(len(matrix[i])):
            if 2989 <= j <= 5383:
                i_temp1 = int(255 * ((matrix[i][j] - minn_in_line_v) / (maxx - minn_in_line_v)))
                j_temp1 = int(255 * ((matrix[i][j] - minn_in_column_v) / (maxx - minn_in_column_v)))
                r = pix[i_temp1, j_temp1][0]  # find value of red
                g = pix[i_temp1, j_temp1][1]  # green
                b = pix[i_temp1, j_temp1][2]  # blue
                matrix_3d[i][j] = [r, g, b]
            else:
                matrix_3d[i][j] = [matrix[i][j] * 255, matrix[i][j] * 255, matrix[i][j] * 255]
    for i in range(733):
        for j in range(len(matrix[i])):
            draw.point((j, i), (matrix_3d[i][j][0], matrix_3d[i][j][1], matrix_3d[i][j][2]))
    return matrix_3d, matrix


def draw(img, start_latlon, end_latlon):
    yaw = 0.
    vscale = 1
    hscale = 1

    # Compute the great-circle distance between two points
    # The units of all input and output parameters are radians
    def distance(lat1, lon1, lat2, lon2):

        delta_lon = lon2 - lon1

        cos_central_angle = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(delta_lon)

        if cos_central_angle < -1:
            cos_central_angle = -1

        if cos_central_angle > 1:
            cos_central_angle = 1

        return np.arccos(cos_central_angle)

    height = len(img)
    y_res = distance(*start_latlon, *end_latlon) / height / vscale
    x_res = 0.0005 / hscale

    # Compute azimuth of line between two points
    # The angle between the line segment defined by the points (`lat1`,`lon1`) and (`lat2`,`lon2`) and the North
    # The units of all input and output parameters are radians
    def azimuth(lat1, lon1, lat2, lon2):

        delta_lon = lon2 - lon1

        return np.arctan2(np.sin(delta_lon), np.cos(lat1) * np.tan(lat2) - np.sin(lat1) * np.cos(delta_lon))

    ref_az = azimuth(*start_latlon, *end_latlon)

    def latlon_to_rel_px(latlon):
        az = azimuth(*start_latlon, *latlon)
        B = az - ref_az

        c = distance(*latlon, *start_latlon)

        if c < -np.pi / 3:
            c = -np.pi / 3

        if c > np.pi / 3:
            c = np.pi / 3

        a = np.arctan(np.cos(B) * np.tan(c))
        b = np.arcsin(np.sin(B) * np.sin(c))

        x = -b / x_res

        # Add the yaw correction value
        # Should be calculating sin(yaw) * x but yaw is always a small value
        y = a / y_res + yaw * x

        return (x, y)

    def draw_line(latlon1, latlon2, r, g, b, a):
        # Convert latlon to (x, y)
        (x1, y1) = latlon_to_rel_px(latlon1)
        (x2, y2) = latlon_to_rel_px(latlon2)

        f = interpolate.interp1d((x1, x2), (y1, y2))
        xar = np.arange(x1, x2)
        dlimg = len(img[0]) / 2080
        bounds_1 = int(dlimg * 456)  # 456
        bounds_2 = int(dlimg * 600)  # 600
        shift_1 = int(dlimg * 539)  # 539
        shift_2 = int(dlimg * 1579)  # 1579
        if (-bounds_1 < x1 < bounds_1 and 0. < y1 < height) or (
                -bounds_2 < x1 < bounds_2 and 0. < y1 < height):
            for x in xar:
                y = f(x)
                if -bounds_1 < x < bounds_1 and 0 < y < height:
                    img[int(y), int(x) + shift_1] = [r, g, b]
                    img[int(y), int(x) + shift_2] = [r, g, b]

    def draw_shape(shpfile, r, g, b):
        reader = shapefile.Reader(shpfile)
        for shape in reader.shapes():
            prev_pt = shape.points[0]
            for pt in shape.points:
                draw_line(
                    (pt[1] / 180. * np.pi, pt[0] / 180. * np.pi),
                    (prev_pt[1] / 180. * np.pi, prev_pt[0] / 180. * np.pi),
                    r, g, b, 0
                )
                prev_pt = pt

    draw_shape("ne_10m_admin_0_countries.shp", 0, 255, 0)
    return img


def draw_image(image, matrix, start_latlon, end_latlon):
    image_new = draw(image, start_latlon, end_latlon)
    matrix_1 = []
    for i in range(733):
        data_is = matrix[i][2989:5384]
        matrix_1.append(data_is)
    matrix_1 = np.array(matrix_1)
    pil_image = Image.fromarray(matrix_1 * 255).convert('RGB')
    pil_image_all = Image.fromarray(matrix * 255).convert('RGB')
    draw_all = ImageDraw.Draw(pil_image_all)
    draw_area = ImageDraw.Draw(pil_image)
    for i in range(733):
        for j in range(2989, 5384):
            draw_area.point((j - 2989, i), (image_new[i][j][0], image_new[i][j][1], image_new[i][j][2]))
    pil_image.save('with_continents_2.png')
    for i in range(len(image_new)):
        for j in range(len(image_new[0])):
            draw_all.point((j, i), (image_new[i][j][0], image_new[i][j][1], image_new[i][j][2]))
    pil_image_all.save('with_continents_2_all.png')


data, fs = sf.read('190422.WAV')
rate_grafic(data)
cut = 4
x = np.arange(0, len(data), 1)
smooth_data = hilbert(data)

latlon_r, lonlat_g = find_latlon()

proj = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
coord_of_start = proj.transform(lonlat_g[0][0], lonlat_g[0][1])
coord_of_end = proj.transform(lonlat_g[1][0], lonlat_g[1][1])
coordinates_file = open('coordinates_of_start_end', 'w')
try:
    for coord in coord_of_start:
        coordinates_file.write(str(coord) + '\t')
    coordinates_file.write('\n')
    for coord in coord_of_end:
        coordinates_file.write(str(coord) + '\t')
except Exception:
    print('error with write in coordinates file')
else:
    print('coordinates write in file')
finally:
    coordinates_file.close()
matrix = norm_straighter(smooth_data)
matrix = normalize(matrix)
matrix_after_color, matrix = paint_image(matrix)
draw_image(matrix_after_color, matrix, latlon_r[0], latlon_r[1])
