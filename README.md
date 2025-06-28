# python-geomorphons
A Python implementation of the Geomorphons algorithm for landform classification from DEMs.

### Geomorphons

A Python port of the `WhiteboxTools` geomorphon landform classification algorithm, adapted for raster processing using `numpy` and `rasterio`. This tool classifies each DEM cell into one of ten basic geomorphic forms (e.g., ridge, valley, slope) or a unique ternary signature.

* https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html#Geomorphons

---

**Geomorphons** are local landform elements identified by comparing elevation angles in 8 radial directions. Based on the work of Jasiewicz & Stepinski (2013), each cell in a digital elevation model (DEM) is evaluated for shape context and assigned a classification.

* https://www.sciencedirect.com/science/article/abs/pii/S0169555X12005028

---

This tool supports:
* 10-class forms (e.g., peak, spur, valley)
* Global ternary codes (GTCs)


| Class ID | Landform    |
|----------|-------------|
| 1        | Flat        |
| 2        | Peak        |
| 3        | Ridge       |
| 4        | Shoulder    |
| 5        | Spur        |
| 6        | Slope       |
| 7        | Hollow      |
| 8        | Footslope   |
| 9        | Valley      |
| 10       | Pit         |

---
**Functions**

- Full raster classification from any DEM (GeoTIFF)
  
```bash
python -W ignore geomorphons.py \
    --dem input_dem.tif \
    --output geomorphons_output.tif \
    --search 50 \
    --threshold 0 \
    --fdist 0 \
    --skip 0 \
    --forms \
    --residuals
```
```
geomorphons(
        'input DEM path',
        'output tif path',
        search=50,
        threshold_deg=0,
        fdist=0,
        skip=0,
        forms=True,
        residuals=False
    )
```
    
- Single-cell query function (`geomorphon_of_cell`) for interactive or custom tools
  
```
landform = geomorphon_of_cell(
         row, col,
         dem= 'input DEM path',
         search=50,
         threshold_deg=0,
         fdist=0,
         skip=0,
         forms=True
     )
```
