def extract_metadata(path):
    # Open each image.
    with open(path, 'rb') as image_file:
        my_image = Image(image_file)

    # Convert coordinates from D/M/S notation into decimal.
    convertedLat = (my_image.gps_latitude[0] + (my_image.gps_latitude[1] / 60) + (my_image.gps_latitude[2] / 3600))
    if my_image.gps_latitude_ref == 'S':
        convertedLat = -convertedLat
    convertedLon = (my_image.gps_longitude[0] + (my_image.gps_longitude[1] / 60) + (my_image.gps_longitude[2] / 3600))
    if my_image.gps_longitude_ref == 'W':
        convertedLon = -convertedLon

    return convertedLat, convertedLon
