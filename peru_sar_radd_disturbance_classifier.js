// ============================================================
// PERUVIAN AMAZON — ILLEGAL ACTIVITY DETECTION PIPELINE
// SAR-based disturbance classification with RADD alert integration
//
// Classifies RADD forest disturbance alerts inside protected areas
// using Sentinel-1 SAR features and a Random Forest classifier.
//
// Output classes:
//   0 = Intact forest
//   1 = Historical disturbance
//   2 = Active disturbance (includes freshly cleared — spectrally identical at 100m)
//   3 = Coca cultivation
//
// Output threat levels:
//   LOW      → intact forest or RADD false positive
//   MEDIUM   → historical disturbance / abandoned site
//   HIGH     → active disturbance (moderate confidence) or coca cultivation
//   CRITICAL → active disturbance (high confidence, active_proportion ≥ 0.7)
//
// Study area: Madre de Dios focus region, Peruvian Amazon
// SAR data:   Sentinel-1 GRD, descending orbit, May–Jul 2025
// ============================================================

// ============================================================
// 1. SETUP — Peru Amazon region and masking layers
// ============================================================
var countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017');
var peru = countries.filter(ee.Filter.eq('country_na', 'Peru'));
var peruGeom = peru.geometry();

var dem = ee.Image('USGS/SRTMGL1_003');
var slope = ee.Terrain.slope(dem);
var gfc = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');

// Amazon mask: retain lowland areas (<1000m, <20° slope) with
// tree cover ≥10% or documented historical forest loss
var amazonMask = dem.lt(1000).and(slope.lt(20))
  .and(gfc.select('treecover2000').gte(10).or(gfc.select('loss').eq(1)));

// ============================================================
// 2. BUILD SAR FEATURE STACK
// Sentinel-1 GRD — descending orbit, dual polarisation (VV/VH)
// Gamma-nought correction applied using local incidence angle
// 15 features: backscatter, VV/VH ratio, GLCM texture (VV + VH)
// ============================================================
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(peruGeom)
  .filterDate('2025-05-01', '2025-07-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select(['VV', 'VH', 'angle']);

// Gamma-nought incidence angle correction
var s1Median   = s1.median().clip(peruGeom);
var angleRad   = s1Median.select('angle').multiply(Math.PI / 180);
var correction = angleRad.cos().log10().multiply(10);
var gammaNought = s1Median.select(['VV', 'VH']).subtract(correction);

// Polarisation ratio — sensitive to surface roughness and structure
var vvvhRatio = gammaNought.select('VV')
  .subtract(gammaNought.select('VH')).rename('VV_VH_ratio');

// Integer quantisation for GLCM computation
var vvScaled = gammaNought.select('VV').add(30).multiply(8).toInt();
var vhScaled = gammaNought.select('VH').add(35).multiply(6).toInt().rename('VH_int');

// GLCM texture features — window size 5 pixels
// Captures spatial structure that separates disturbance types
var glcmVV = vvScaled.glcmTexture({size: 5});
var glcmVH = vhScaled.glcmTexture({size: 5});

// Assemble final 15-band feature stack
var featureStack = gammaNought.select(['VV', 'VH'])
  .addBands(vvvhRatio)
  .addBands(glcmVV.select(['VV_contrast', 'VV_diss', 'VV_idm', 'VV_asm', 'VV_ent', 'VV_corr']))
  .addBands(glcmVH.select(['VH_int_contrast', 'VH_int_diss', 'VH_int_idm', 'VH_int_asm', 'VH_int_ent', 'VH_int_corr']))
  .updateMask(amazonMask)
  .clip(peruGeom);

var bandNames = featureStack.bandNames();
print('SAR bands:', bandNames);

// ============================================================
// 3. CONVERT TRAINING POLYGONS TO FEATURE COLLECTIONS
//
// Polygon geometry types differ by import:
//   MultiPolygon  → use .coordinates() + ee.Geometry.Polygon()
//   GeometryCollection → use .geometries() + ee.Geometry()
//
// Polygon ID ranges prevent collisions across classes:
//   0–999    = Intact forest
//   1000–1999 = Historical disturbance
//   2000–2999 = Active disturbance
//   3000–3999 = Coca cultivation
//   4000–4999 = Freshly cleared (merged into active at training)
// ============================================================

// Intact forest — MultiPolygon
var forestCoords = IntactForest.coordinates();
var fc_forest = ee.FeatureCollection(
  forestCoords.zip(ee.List.sequence(0, forestCoords.size().subtract(1)))
  .map(function(item) {
    var pair = ee.List(item);
    return ee.Feature(ee.Geometry.Polygon(pair.get(0)))
      .set('poly_id', ee.Number(pair.get(1)).add(0));
  })
);

// Historical disturbance — old mining scars, stable ponds, weathered tailings
var historicalGeoms = HistoricalDisturbance.geometries();
var fc_historical = ee.FeatureCollection(
  historicalGeoms.zip(ee.List.sequence(0, historicalGeoms.size().subtract(1)))
  .map(function(item) {
    var pair = ee.List(item);
    return ee.Feature(ee.Geometry(pair.get(0)))
      .set('poly_id', ee.Number(pair.get(1)).add(1000));
  })
);

// Active disturbance — operating mines, turbid water, fresh bare soil
var activeGeoms = ActiveDisturbance.geometries();
var fc_active = ee.FeatureCollection(
  activeGeoms.zip(ee.List.sequence(0, activeGeoms.size().subtract(1)))
  .map(function(item) {
    var pair = ee.List(item);
    return ee.Feature(ee.Geometry(pair.get(0)))
      .set('poly_id', ee.Number(pair.get(1)).add(2000));
  })
);

// Coca cultivation — uniform low bushy canopy with visible row structure
var cocaGeoms = CocaCultivation.geometries();
var fc_coca = ee.FeatureCollection(
  cocaGeoms.zip(ee.List.sequence(0, cocaGeoms.size().subtract(1)))
  .map(function(item) {
    var pair = ee.List(item);
    return ee.Feature(ee.Geometry(pair.get(0)))
      .set('poly_id', ee.Number(pair.get(1)).add(3000));
  })
);

// Freshly cleared forest — recent felling with debris present
// NOTE: Classified as label 2 (active disturbance) at training time
// because SAR backscatter from fresh debris is indistinguishable
// from active mining at 100m resolution (producer accuracy 0% when
// treated as a separate class). Merged here rather than deleted to
// maximise active disturbance training sample size.
var freshCoords = FreshlyCleared.coordinates();
var fc_fresh = ee.FeatureCollection(
  freshCoords.zip(ee.List.sequence(0, freshCoords.size().subtract(1)))
  .map(function(item) {
    var pair = ee.List(item);
    return ee.Feature(ee.Geometry.Polygon(pair.get(0)))
      .set('poly_id', ee.Number(pair.get(1)).add(4000));
  })
);

print('Intact forest polygons:',          fc_forest.size());
print('Historical disturbance polygons:', fc_historical.size());
print('Active disturbance polygons:',     fc_active.size());
print('Coca cultivation polygons:',       fc_coca.size());
print('Freshly cleared polygons:',        fc_fresh.size());

// ============================================================
// 4. SAMPLE + BALANCE
// Each class sampled independently then capped to the size of
// the smallest class — prevents class imbalance from dominating
// the Random Forest decision boundaries.
// Coca sampled at 30m to extract sufficient pixels from small plots.
// All other classes sampled at 100m (native SAR resolution).
// ============================================================
var forestAll = featureStack.sampleRegions({
  collection: fc_forest.map(function(f) { return f.set('label', 0); }),
  properties: ['label', 'poly_id'], scale: 100, tileScale: 4
}).randomColumn('rand', 42).sort('rand');

var historicalAll = featureStack.sampleRegions({
  collection: fc_historical.map(function(f) { return f.set('label', 1); }),
  properties: ['label', 'poly_id'], scale: 100, tileScale: 4
}).randomColumn('rand', 42).sort('rand');

// Merge freshly cleared into active disturbance before sampling
var fc_active_combined = fc_active.merge(
  fc_fresh.map(function(f) { return f.set('label', 2); })
);

var activeAll = featureStack.sampleRegions({
  collection: fc_active_combined.map(function(f) { return f.set('label', 2); }),
  properties: ['label', 'poly_id'], scale: 100, tileScale: 4
}).randomColumn('rand', 42).sort('rand');

// Coca sampled at 30m — plots are small and yield few pixels at 100m
var cocaAll = featureStack.sampleRegions({
  collection: fc_coca.map(function(f) { return f.set('label', 3); }),
  properties: ['label', 'poly_id'], scale: 30, tileScale: 4
}).randomColumn('rand', 42).sort('rand');

// Cap all classes to the size of the smallest — automatic rebalancing
var classCap = forestAll.size()
  .min(historicalAll.size())
  .min(activeAll.size())
  .min(cocaAll.size());

print('Class cap (smallest class sets limit):', classCap);

var training = forestAll.limit(classCap)
  .merge(historicalAll.limit(classCap))
  .merge(activeAll.limit(classCap))
  .merge(cocaAll.limit(classCap));

print('Intact forest pixels:',          forestAll.limit(classCap).size());
print('Historical disturbance pixels:', historicalAll.limit(classCap).size());
print('Active disturbance pixels:',     activeAll.limit(classCap).size());
print('Coca cultivation pixels:',       cocaAll.limit(classCap).size());
print('Total balanced training pixels:', training.size());

// ============================================================
// 5. POLYGON-LEVEL TRAIN/TEST SPLIT
// Split on polygon ID rather than individual pixels to prevent
// spatial autocorrelation leaking between train and test sets.
// Pixels from the same polygon only appear in one partition.
// 75% train / 25% test split applied to polygon IDs.
// ============================================================
var polyIds       = training.distinct('poly_id');
var polyIdsRandom = polyIds.randomColumn('rand', 42);
var trainPolyIds  = polyIdsRandom.filter(ee.Filter.lt('rand', 0.75));
var testPolyIds   = polyIdsRandom.filter(ee.Filter.gte('rand', 0.75));

var polyJoin    = ee.Join.saveFirst('split_info');
var trainFilter = ee.Filter.equals({leftField: 'poly_id', rightField: 'poly_id'});

var trainPixels = polyJoin.apply(training, trainPolyIds, trainFilter);
var testPixels  = polyJoin.apply(training, testPolyIds,  trainFilter);

print('Train polygons:', trainPolyIds.size(), '| Test polygons:', testPolyIds.size());
print('Train pixels:',  trainPixels.size(),  '| Test pixels:',  testPixels.size());

// ============================================================
// 6. EVALUATION MODEL — trained on split for accuracy reporting
// 500 trees, minimum leaf population 2, fixed seed for
// reproducibility. Evaluated on held-out test polygons only.
// ============================================================
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 500,
  minLeafPopulation: 2,
  seed: 42
}).train({
  features: trainPixels,
  classProperty: 'label',
  inputProperties: bandNames
});

var validated   = testPixels.classify(classifier);
var errorMatrix = validated.errorMatrix('label', 'classification');

print('=== MODEL EVALUATION (polygon-level split) ===');
print('Confusion matrix:', errorMatrix);
print('Overall accuracy:', errorMatrix.accuracy());
print('Kappa:',           errorMatrix.kappa());
print('Producers accuracy:', errorMatrix.producersAccuracy());
print('Consumers accuracy:', errorMatrix.consumersAccuracy());

// ============================================================
// 7. DEPLOYMENT MODEL — retrained on full dataset
// Separate from the evaluation model to maximise training data
// for production classification. Same hyperparameters as above.
// ============================================================
var finalClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 500,
  minLeafPopulation: 2,
  seed: 42
}).train({
  features: training,
  classProperty: 'label',
  inputProperties: bandNames
});

// ============================================================
// 8. RADD FOREST DISTURBANCE ALERTS
// Source: Radar for Detecting Deforestation (RADD), WUR
// Filtered to: South America, Alert ≥ 2 (confirmed disturbance),
// Date ≥ 2600 (recent alerts only), within Amazon mask
// ============================================================
var radd = ee.ImageCollection('projects/radar-wur/raddalert/v1');

var latestRADD = ee.Image(radd
  .filterMetadata('layer', 'contains', 'alert')
  .filterMetadata('geography', 'equals', 'sa')
  .sort('system:time_end', false)
  .first()
);

var raddAlert = latestRADD.select('Alert');
var raddDate  = latestRADD.select('Date');

var recentRADD = raddAlert.gte(2)
  .and(raddDate.gte(2600))
  .selfMask();

var recentRADDPeru = recentRADD.updateMask(amazonMask).clip(peruGeom);
print('RADD alert layer ready');

// ============================================================
// 9. PROTECTED AREA BOUNDARIES
// Source: WDPA (World Database on Protected Areas)
// Used to identify alerts occurring inside protected zones
// where any disturbance is by definition illegal
// ============================================================
var wdpa          = ee.FeatureCollection('WCMC/WDPA/current/polygons');
var peruProtected = wdpa.filterBounds(peruGeom);
var protectedMask = ee.Image().byte().paint(peruProtected, 1).selfMask();

// ============================================================
// 10. FOCUS REGION
// Primary study area: known illegal gold mining corridor
// RADD alerts restricted to protected areas within this extent
// Alert centroids extracted at 500m scale for processing
// ============================================================
var focusRegion = ee.Geometry.Rectangle([-73.5, -11.5, -70.5, -9.5]);

var raddInProtectedFocus = recentRADDPeru.updateMask(protectedMask).clip(focusRegion);

var alertPatches = raddInProtectedFocus.reduceToVectors({
  geometry: focusRegion,
  scale: 500,
  geometryType: 'centroid',
  eightConnected: true,
  maxPixels: 1e9,
  bestEffort: true,
  tileScale: 4
});

print('RADD alerts in protected areas:', alertPatches.size());

// ============================================================
// 11. ALERT SITE CLASSIFICATION — majority vote
// Each alert centroid buffered 400m, SAR pixels sampled within
// buffer and classified individually. Site-level classification
// assigned by majority vote across all sampled pixels.
// active_proportion measures confidence: fraction of pixels
// voting active disturbance — drives CRITICAL vs HIGH threshold.
// Processing capped at 200 sites due to GEE runtime limits.
// ============================================================
var alertsNumbered = alertPatches.limit(200).map(function(f) {
  return f.set('alert_id', f.id());
});

var alertBuffered = alertsNumbered.map(function(f) {
  return f.setGeometry(f.geometry().buffer(400));
});

var alertSamples = featureStack.sampleRegions({
  collection: alertBuffered,
  properties: ['alert_id'],
  scale: 100,
  tileScale: 8
});

var pixelClassified = alertSamples.classify(finalClassifier);

var alertSiteResults = alertsNumbered.map(function(alert) {
  var aid     = alert.get('alert_id');
  var pixels  = pixelClassified.filter(ee.Filter.eq('alert_id', aid));
  var nPixels = pixels.size();

  // Count pixels per class within the alert buffer
  var nForest     = pixels.filter(ee.Filter.eq('classification', 0)).size();
  var nHistorical = pixels.filter(ee.Filter.eq('classification', 1)).size();
  var nActive     = pixels.filter(ee.Filter.eq('classification', 2)).size();
  var nCoca       = pixels.filter(ee.Filter.eq('classification', 3)).size();

  var maxCount = nForest.max(nHistorical).max(nActive).max(nCoca).max(nFresh);

  // Majority vote — ties broken in favour of higher threat class
  var voted = ee.Algorithms.If(nActive.eq(maxCount),     2,
              ee.Algorithms.If(nFresh.eq(maxCount),      4,
              ee.Algorithms.If(nCoca.eq(maxCount),       3,
              ee.Algorithms.If(nHistorical.eq(maxCount), 1, 0))));

  // active_proportion: key confidence metric driving threat level
  var activeProp = ee.Number(nActive).divide(ee.Number(nPixels).max(1));

  return alert.set({
    'classification':    voted,
    'n_pixels':          nPixels,
    'n_forest':          nForest,
    'n_historical':      nHistorical,
    'n_active':          nActive,
    'n_coca':            nCoca,
    'active_proportion': activeProp
  });
});

// ============================================================
// 12. THREAT LEVEL ASSIGNMENT + ACTIVITY LABELLING
// Regional lookup zones assign probable activity type based on
// documented illegal activity patterns by region (UNODC / SERNANP).
// Threat level combines SAR classification with active_proportion
// to produce an operationally meaningful priority score.
// NOTE: purpose is prioritisation, not legal determination.
// ============================================================
var madreDeDios    = ee.Geometry.Rectangle([-72.0, -13.5, -68.5, -11.0]);
var ucayaliHuanuco = ee.Geometry.Rectangle([-76.0, -11.0, -73.0,  -7.5]);
var loreto         = ee.Geometry.Rectangle([-77.5,  -7.5, -72.0,  -1.0]);
var sanMartin      = ee.Geometry.Rectangle([-78.0,  -8.0, -75.5,  -5.0]);

var alertsFinal = alertSiteResults.map(function(f) {
  var cls        = ee.Number(f.get('classification'));
  var geom       = f.geometry();
  var activeProp = ee.Number(f.get('active_proportion'));

  var inMdD = madreDeDios.contains(geom);
  var inUcH = ucayaliHuanuco.contains(geom);
  var inLor = loreto.contains(geom);
  var inSM  = sanMartin.contains(geom);

  // Geographic region label
  var region = ee.Algorithms.If(inMdD, 'Madre de Dios',
               ee.Algorithms.If(inUcH, 'Ucayali / Huanuco',
               ee.Algorithms.If(inLor, 'Loreto',
               ee.Algorithms.If(inSM,  'San Martin / Amazonas', 'Other Amazon'))));

  // Dominant illegal activity type for this region
  // Based on UNODC monitoring and SERNANP protected area reports
  var regionalActivity = ee.Algorithms.If(inMdD, 'Illegal gold mining',
                         ee.Algorithms.If(inUcH, 'Illegal logging / coca cultivation',
                         ee.Algorithms.If(inLor, 'Illegal logging',
                         ee.Algorithms.If(inSM,  'Narcotics-linked deforestation',
                         'Unknown illegal activity'))));

  // Threat level logic:
  // CRITICAL = active disturbance, active_proportion ≥ 0.7
  // HIGH     = active disturbance (0.5–0.7), or coca, or freshly cleared
  // MEDIUM   = historical disturbance (abandoned site)
  // LOW      = intact forest or RADD false positive
  var threatLevel = ee.Algorithms.If(cls.eq(2).and(activeProp.gte(0.7)), 'CRITICAL',
                    ee.Algorithms.If(cls.eq(2).and(activeProp.gte(0.5)), 'HIGH',
                    ee.Algorithms.If(cls.eq(4), 'HIGH',
                    ee.Algorithms.If(cls.eq(3), 'HIGH',
                    ee.Algorithms.If(cls.eq(1), 'MEDIUM', 'LOW')))));

  // Human-readable label for operational use
  var activityLabel = ee.Algorithms.If(cls.eq(2).and(activeProp.gte(0.7)),
      ee.String('HIGH CONFIDENCE: Active ').cat(ee.String(regionalActivity)),
    ee.Algorithms.If(cls.eq(2).and(activeProp.gte(0.5)),
      ee.String('MODERATE CONFIDENCE: Likely active ').cat(ee.String(regionalActivity)),
    ee.Algorithms.If(cls.eq(4),
      ee.String('FRESHLY CLEARED: Recent forest removal — possible ').cat(ee.String(regionalActivity)),
    ee.Algorithms.If(cls.eq(3),
      'COCA CULTIVATION detected — illegal narcotics production in protected area',
    ee.Algorithms.If(cls.eq(1),
      ee.String('HISTORICAL DISTURBANCE: Abandoned site — possible ').cat(ee.String(regionalActivity)),
      'LOW RISK: Intact forest or RADD false positive')))));

  return f.set({
    'region':         region,
    'activity_label': activityLabel,
    'threat_level':   threatLevel
  });
});

// ============================================================
// 13. RESULTS SUMMARY
// ============================================================
var critical = alertsFinal.filter(ee.Filter.eq('threat_level', 'CRITICAL'));
var high     = alertsFinal.filter(ee.Filter.eq('threat_level', 'HIGH'));
var medium   = alertsFinal.filter(ee.Filter.eq('threat_level', 'MEDIUM'));
var low      = alertsFinal.filter(ee.Filter.eq('threat_level', 'LOW'));

print('');
print('============================================');
print(' ILLEGAL ACTIVITY SCREENING RESULTS');
print('============================================');
print('Total RADD alert sites analysed:', alertsFinal.size());
print('CRITICAL:', critical.size());
print('HIGH:',     high.size());
print('MEDIUM:',   medium.size());
print('LOW:',      low.size());

// ============================================================
// 14. EXPORT
// Two outputs:
//   GeoTIFF — wall-to-wall classification + threat raster + protected area mask
//   CSV     — per-site classification table for GIS review
// ============================================================
var classifiedImage = featureStack.classify(finalClassifier)
  .clip(focusRegion).toByte();

// Paint threat levels onto raster — higher threat overwrites lower
var threatImage = ee.Image(0).byte().clip(focusRegion).rename('threat_level');
threatImage = threatImage
  .paint(medium.map(function(f)   { return f.buffer(500); }), 1)
  .paint(high.map(function(f)     { return f.buffer(500); }), 2)
  .paint(critical.map(function(f) { return f.buffer(500); }), 3)
  .selfMask();

// Three-band Cloud-Optimized GeoTIFF
var exportImage = classifiedImage.rename('classification')
  .addBands(threatImage.rename('threat_level'))
  .addBands(protectedMask.rename('protected_area').unmask(0).toByte().clip(focusRegion));

var today = '2026_04_20';

Export.image.toDrive({
  image:       exportImage,
  description: 'disturbance_classification' + today,
  folder:      'peru_detections',
  region:      focusRegion,
  scale:       100,
  crs:         'EPSG:4326',
  maxPixels:   1e10,
  fileFormat:  'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Export.table.toDrive({
  collection:  alertsFinal,
  description: 'alert_sites' + today,
  folder:      'peru_detections',
  fileFormat:  'CSV'
});

print('');
print('=== EXPORTS QUEUED — check Tasks tab ===');
print('COG Band 1 (classification): 0=intact forest, 1=historical disturbance, 2=active disturbance, 3=coca cultivation');
print('COG Band 2 (threat_level):   0=none, 1=MEDIUM, 2=HIGH, 3=CRITICAL');
print('COG Band 3 (protected_area): 0=outside protected area, 1=inside protected area');
