# Analysis

After estimating the Food Environment Nutrient Density (*FEND*) on all available counties (or census tracts), run the Rmd file and python files included in this folder to perform analysis on how healthy food accessibility, characterized with FEND, is affected by socioeconomic and spatial differences and how it is strongly associated with diet-related disease in the area.  

Knit R markdown file to html that inlcudes linear modeling for county-level analysis:
```
Rscript -e "rmarkdown::render('linear_model_analysis_county.Rmd', output_format='html_document', output_file='linear_model_analysis_county.html')"
```

Produce results, figures, and maps included in the manuscript. Choose nutrient density score that the FEND dataframe is based on and threshold to apply for the analysis:
```
python results_figures.py RRR 50
python maps.py RRR 50
```

