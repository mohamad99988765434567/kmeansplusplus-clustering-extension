# K-means++ Clustering Extension

This project implements the K-means++ initialization and clustering algorithm using a combination of Python and C for better performance. The Python script handles data input and initial centroid selection, while the main clustering logic is done in C.

## Project Structure

- `kmeans_pp.py`: Reads command-line arguments, loads and processes data, performs K-means++ initialization, and calls the C extension.
- `kmeansmodule.c`: C extension that runs the main K-means algorithm.
- `setup.py`: Script to build the C extension.
- `bonus.py` (optional): Uses the Elbow method to find the best number of clusters on the Iris dataset.

## How to Run

### Step 1: Build the C extension
```bash
python3 setup.py build_ext --inplace
```

### Step 2: Run the program
```bash
python3 kmeans_pp.py <K> [max_iter] <eps> <input_file_1> <input_file_2>
```

Example:
```bash
python3 kmeans_pp.py 3 100 0.01 input1.txt input2.txt
```

## Output Format
- First line: Indices of initial centroids selected by K-means++ (comma-separated)
- Following lines: Final centroids, each line contains one centroid with values rounded to 4 decimal places

## Requirements
- Python 3.x
- numpy
- A C compiler (for the extension build)

## Notes
- Input files are merged using the first column as a key
- Output values are formatted to 4 decimal places
- Make sure to free memory properly in the C code

## Bonus (Optional)
Run `bonus.py` to generate a plot (`elbow.png`) showing how inertia changes for different values of K (from 1 to 10).

