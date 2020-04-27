from pyspark import SparkContext
import csv
import geopandas as gpd
import fiona
import fiona.crs
import shapely
import sys


def createIndex(geojson):
    '''
    This function takes in a shapefile path, and return:
    (1) index: an R-Tree based on the geometry data in the file
    (2) zones: the original data of the shapefile
    
    Note that the ID used in the R-tree 'index' is the same as
    the order of the object in zones.
    '''
    import rtree
    import fiona.crs
    import geopandas as gpd
    zones = gpd.read_file(geojson).to_crs(fiona.crs.from_epsg(2263))
    index = rtree.Rtree()
    for idx,geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return (index, zones)


def findNeighborhoods(p, index, zones):
    '''
    findZone returned the ID of the shape (stored in 'zones' with
    'index') that contains the given point 'p'. If there's no match,
    None will be returned.
    '''
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return idx
    return None


def findBoroughs(p, index, zones):
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return idx
    return None


def processTrips(pid, records):
    '''
    Our aggregation function that iterates through records in each
    partition, checking whether we could find a zone that contain
    the pickup location.
    '''
    import csv
    import pyproj
    import shapely.geometry as geom
    
    if pid==0:
        next(records)
    reader = csv.reader(records)
    counts = {}
    # Create an R-tree index
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    
    s_index, s_zones = createIndex('boroughs.geojson')    
    e_index, e_zones = createIndex('neighborhoods.geojson')
    
    for row in reader:
        try:
            s_p = geom.Point(proj(float(row[5]), float(row[6])))
            e_p = geom.Point(proj(float(row[9]), float(row[10])))
        
        except:
            continue
        
        borough = findBoroughs(s_p, s_index, s_zones)
        neighborhood = findNeighborhoods(e_p, e_index, e_zones)
        
        if borough and neighborhood:
            key = (borough, neighborhood)
            counts[key] = counts.get(key, 0) + 1
    return counts.items()


def organize(records):
    res = {}
    for record in records:
        if record[0] not in res:
            res[record[0]] = []
        res[record[0]].append(record[1])
    return res.items()


def mapper(line):
    b, n = line[0], line[1]
    n.sort(key=lambda x: x[1], reverse=True)
    n = n[:3]
    return b, n[0][0], n[0][1], n[1][0], n[1][1], n[2][0], n[2][1]


if __name__ == "__main__":
    
    sc = SparkContext()
    
    boroughs_geojson = 'boroughs.geojson'
    neighborhoods_geojson = 'neighborhoods.geojson'
    input_file = sys.argv[1]
    output = sys.argv[2]
    
    boroughs = gpd.read_file(boroughs_geojson).to_crs(fiona.crs.from_epsg(2263))
    neighborhoods = gpd.read_file(neighborhoods_geojson).to_crs(fiona.crs.from_epsg(2263))
    
    borough_list = list(boroughs['boro_name'])
    neighborhood_list = list(neighborhoods['neighborhood'])
    
    rdd = sc.textFile(input_file)
    counts = rdd.filter(lambda row: len(row)>9) \
                .mapPartitionsWithIndex(processTrips) \
                .reduceByKey(lambda x,y: x+y) \
                .map(lambda x: ((borough_list[x[0][0]]), (neighborhood_list[x[0][1]], x[1]))) \
                .mapPartitions(organize) \
                .reduceByKey(lambda x,y: x+y) \
                .map(mapper) \
                .sortByKey() \
                .collect()
    print(counts)
