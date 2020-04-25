import csv
import itertools
import sys
from pyspark import SparkContext

def createIndex(shapefile):
    import rtree
    import fiona.crs
    import geopandas as gpd
    zones = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(2263))
    index = rtree.Rtree()
    for idx,geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return (index, zones)

def findBoro(p, index, zones):
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return str(zones.CITY[idx]).split('-')[1]

def findZone(p, index, zones):
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return str(zones.NAME[idx])

def processTrips(pid, records):
    '''
    Match each record with its starting borough and its destination zone
    '''
    import csv
    import pyproj
    import shapely.geometry as geom
    
    # Create an R-tree index
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    index, zones = createIndex('ZillowNeighborhoods-NY/ZillowNeighborhoods-NY.shp')    
    
    # Skip the header
    if pid==0:
        next(records)
    reader = csv.reader(records)
    
    for row in reader:
        if 'NULL' in row[2:6]: 
            continue # ignore trips without locations
            
        p1 = geom.Point(proj(float(row[3]), float(row[2]))) # get the pick up point to find the borough
        p2 = geom.Point(proj(float(row[5]), float(row[4]))) # get the destination to find the zone

        borough = findBoro(p1, index, zones)
        zone = findZone(p2, index, zones)
        if borough!=None and zone!=None:
            yield (borough, zone), 1

def toCSV(_, records):
    for (boro, top1, num1, top2, num2, top3, num3) in records:
        yield ','.join((boro, top1, str(num1), top2, str(num2), top3, str(num3)))

if __name__=='__main__':
    sc = SparkContext()
    sc.textFile(sys.argv[1])     .mapPartitionsWithIndex(processTrips)     .reduceByKey(lambda x,y: x+y)     .map(lambda x: (x[0][0],x[0][1],x[1]))     .sortBy(lambda x: -x[2])     .map(lambda x: (x[0],(x[1],x[2])))     .reduceByKey(lambda x,y: x+y)     .sortByKey()     .map(lambda x: (x[0],x[1][0],x[1][1],x[1][2],x[1][3],x[1][4],x[1][5]))     .mapPartitionsWithIndex(toCSV)     .saveAsTextFile(sys.argv[2])

