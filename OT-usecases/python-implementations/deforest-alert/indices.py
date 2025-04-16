from torchgeo.transforms import AppendNDVI, AppendNDRE, AppendGNDVI

INDICE_MAP = {'ndvi':AppendNDVI, 'ndre':AppendNDRE, 'gndvi':AppendGNDVI}



# PROBLEM -> Phenology.
# Solutions ->
# We can access via past data the months in which phenology remains stable across years.
# Prithvi-FM is trained via lat, long, time and results show that model has incorporated seasonal effects into its representation.