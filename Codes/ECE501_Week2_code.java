// ================================
// 1. Get country boundaries
// ================================

var countries = ee.FeatureCollection('FAO/GAUL/2015/level0');

var germany = countries.filter(
  ee.Filter.eq('ADM0_NAME', 'Germany')
);

var uruguay = countries.filter(
  ee.Filter.eq('ADM0_NAME', 'Uruguay')
);


// ================================
// 2. Combine Germany + Uruguay
// ================================

var studyArea = germany.merge(uruguay);

print('Study Area:', studyArea);


// ================================
// 3. Show Study Area on Map
// ================================

Map.centerObject(studyArea, 3);

Map.addLayer(
  germany,
  {},
  'Germany'
);

Map.addLayer(
  uruguay,
  {},
  'Uruguay'
);

// ================================
// 4. Sentinel-1 collection
// ================================

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(studyArea)
  .filterDate('2025-06-01', '2025-09-30')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains(
    'transmitterReceiverPolarisation', 'VV'
  ))
  .filter(ee.Filter.listContains(
    'transmitterReceiverPolarisation', 'VH'
  ))
  .select(['VV', 'VH']);


// ================================
// 5. Check result
// ================================

print('Sentinel-1 Collection:', s1);
print('Number of images:', s1.size());

// =====================================
// 4. Create Sentinel-1 Composite
// =====================================

var s1Composite = s1.median();


// =====================================
// 5. Display VV
// =====================================

Map.addLayer(
  s1Composite.select('VV'),
  {
    min: -25,
    max: 5
  },
  'Sentinel-1 VV'
);


// =====================================
// 6. Display VH
// =====================================

Map.addLayer(
  s1Composite.select('VH'),
  {
    min: -30,
    max: 0
  },
  'Sentinel-1 VH'
);

// =====================================
// SENTINEL-2 OPTICAL DATA
// =====================================

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(studyArea)
  .filterDate('2025-06-01', '2025-09-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

print('Sentinel-2 Collection:', s2);
print('Number of Sentinel-2 images:', s2.size());

// =====================================
// SENTINEL-2 COMPOSITE
// =====================================

var s2Composite = s2.median();

print('Sentinel-2 Composite:', s2Composite);

// =====================================
// SELECT OPTICAL BANDS
// =====================================

var s2Features = s2Composite.select([
  'B2',
  'B3',
  'B4',
  'B8'
]);

print('Sentinel-2 Features:', s2Features);

// =====================================
// NDVI
// =====================================

var ndvi = s2Composite.normalizedDifference([
  'B8',
  'B4'
]).rename('NDVI');

Map.addLayer(
  ndvi,
  {
    min: -1,
    max: 1
  },
  'NDVI'
);

// =====================================
// FEATURE FUSION
// =====================================

var fusedFeatures = s1Composite
  .select(['VV', 'VH'])
  .addBands(s2Features)
  .addBands(ndvi);

print('Fused Features:', fusedFeatures);

// =====================================
// 6. DYNAMIC WORLD REFERENCE DATA
// =====================================

var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(studyArea)
  .filterDate('2025-06-01', '2025-09-30');

print('Dynamic World Collection:', dw);
print('Dynamic World images:', dw.size());


// =====================================
// 7. Create Dynamic World Composite
// =====================================

var dwComposite = dw.median();


// =====================================
// 8. Select Dynamic World Label
// =====================================

var dwLabel = dwComposite.select('label');


// Dynamic World crop class = 4
var cropMask = dwLabel.eq(4).rename('crop');


// Display crop reference
Map.addLayer(
  cropMask,
  {
    min: 0,
    max: 1
  },
  'Dynamic World Crops'
);

// =====================================
// 6. DYNAMIC WORLD REFERENCE
// =====================================

var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(studyArea)
  .filterDate('2025-06-01', '2025-09-30');

print('Dynamic World images:', dw.size());


// =====================================
// 7. Get most common land-cover label
// =====================================

var dwLabel = dw.select('label').mode();


// =====================================
// 8. Create Crop / Non-Crop Classes
// =====================================

// Dynamic World class 4 = Crops
// Crop     = 1
// Non-crop = 0

var classImage = dwLabel.eq(4)
  .rename('class');


// Display
Map.addLayer(
  classImage,
  {
    min: 0,
    max: 1
  },
  'Crop / Non-Crop Reference'
);

print('Class Image:', classImage);

// =====================================
// 9. Combine Features + Class
// =====================================

var trainingImage = fusedFeatures.addBands(classImage);

print('Training Image:', trainingImage);

// =====================================
// 10. Create Training Samples
// =====================================

var samples = trainingImage.stratifiedSample({
  numPoints: 500,
  classBand: 'class',
  region: studyArea,
  scale: 10,
  seed: 42,
  geometries: true
});

print('Training Samples:', samples);
print('Number of Samples:', samples.size());

Map.addLayer(
  samples,
  {},
  'Training Samples'
);

// =====================================
// 11. Split Training and Testing Data
// =====================================

var samplesWithRandom = samples.randomColumn(
  'random',
  42
);

var trainingSamples = samplesWithRandom.filter(
  ee.Filter.lt('random', 0.7)
);

var testingSamples = samplesWithRandom.filter(
  ee.Filter.gte('random', 0.7)
);

print('Training samples:', trainingSamples.size());
print('Testing samples:', testingSamples.size());

// =====================================
// 12. Random Forest Classifier
// =====================================

var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  seed: 42
});

var featureBands = [
  'VV',
  'VH',
  'B2',
  'B3',
  'B4',
  'B8',
  'NDVI'
];

// =====================================
// 13. Train Classifier
// =====================================

classifier = classifier.train({
  features: trainingSamples,
  classProperty: 'class',
  inputProperties: featureBands
});

print('Trained Classifier:', classifier);

// =====================================
// 14. Classify Study Area
// =====================================

var classified = fusedFeatures
  .select(featureBands)
  .classify(classifier);

Map.addLayer(
  classified,
  {
    min: 0,
    max: 1
  },
  'Crop Classification'
);

// =====================================
// 15. Test the Classifier
// =====================================

var tested = testingSamples.classify(classifier);

var confusionMatrix = tested.errorMatrix(
  'class',
  'classification'
);

print('Confusion Matrix:', confusionMatrix);
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Kappa:', confusionMatrix.kappa());