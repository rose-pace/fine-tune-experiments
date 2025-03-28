#!/bin/bash

cd "$(dirname "$0")"

# Directory containing the RPG files
DOCS_DIR="../local_files/input"
# Output directory for datasets
OUTPUT_DIR="../local_files/output"
# Document type (default to md)
DOC_TYPE="md"
# Number of questions per chunk
QUESTIONS=5
# Number of distractors
DISTRACTORS=3
# Include oracle percentage
P_VALUE=0.8
# Chunk size
CHUNK_SIZE=1024
# Models to use
QG_MODEL="google/flan-t5-large"
COT_MODEL="google/flan-t5-xl"
QA_MODEL="deepset/deberta-v3-large-squad2"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to get file extension
get_extension() {
    filename=$(basename -- "$1")
    echo "${filename##*.}"
}

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --docs-dir)
            DOCS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --questions)
            QUESTIONS="$2"
            shift 2
            ;;
        --distractors)
            DISTRACTORS="$2"
            shift 2
            ;;
        --p)
            P_VALUE="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --qg-model)
            QG_MODEL="$2"
            shift 2
            ;;
        --cot-model)
            COT_MODEL="$2"
            shift 2
            ;;
        --qa-model)
            QA_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to get file extension
get_extension() {
    filename=$(basename -- "$1")
    echo "${filename##*.}"
}

# Check if directory exists
if [ ! -d "$DOCS_DIR" ]; then
    echo "Error: Directory $DOCS_DIR does not exist."
    exit 1
fi

# Create a log file
LOG_FILE="$OUTPUT_DIR/processing_log.txt"
echo "Starting processing at $(date)" > "$LOG_FILE"

# Track completion status
echo "Processing Status:" >> "$LOG_FILE"

# Create a combined JSONL file for all results
COMBINED_JSONL="$OUTPUT_DIR/combined_raft_data.jsonl"
echo "" > "$COMBINED_JSONL"

# Function to process a file
process_file() {
    file="$1"
    filename=$(basename -- "$file")
    filename_no_ext="${filename%.*}"
    extension=$(get_extension "$file")

    # Set doctype based on file extension
    case "$extension" in
        md)
            doctype="md"
            ;;
        txt)
            doctype="txt"
            ;;
        pdf)
            doctype="pdf"
            ;;
        json)
            doctype="json"
            ;;
        *)
            echo "Unsupported file type: $extension for file $filename"
            return 1
            ;;
    esac

    output_path="$OUTPUT_DIR/$filename_no_ext"
    jsonl_path="${output_path}.jsonl"

    echo "Processing $filename (doctype: $doctype)"
    echo "  - Output: $output_path"

    # Log the start of processing
    echo "  $filename: Started at $(date)" >> "$LOG_FILE"

    # Run the script
    python raft_training_data_gen.py \
        --datapath "$file" \
        --output "$output_path" \
        --doctype "$doctype" \
        --questions "$QUESTIONS" \
        --distractors "$DISTRACTORS" \
        --p "$P_VALUE" \
        --chunk_size "$CHUNK_SIZE" \
        --qg-model "$QG_MODEL" \
        --cot-model "$COT_MODEL" \
        --qa-model "$QA_MODEL"

    status=$?
    if [ $status -eq 0 ]; then
        echo "  $filename: Completed successfully at $(date)" >> "$LOG_FILE"

        # Append content to combined JSONL if file exists
        if [ -f "$jsonl_path" ]; then
            cat "$jsonl_path" >> "$COMBINED_JSONL"
            echo "  - Added to combined JSONL file"
        else
            echo "  - Warning: JSONL file not found at $jsonl_path"
        fi
        return 0
    else
        echo "  $filename: Failed with status $status at $(date)" >> "$LOG_FILE"
        return 1
    fi
}

# Process each file in the directory
echo "Starting to process files in $DOCS_DIR"
file_count=0
success_count=0

for file in "$DOCS_DIR"/*; do
    if [ -f "$file" ]; then
        ((file_count++))
        process_file "$file"
        if [ $? -eq 0 ]; then
            ((success_count++))
        fi
    fi
done

# Log completion
echo "Processing complete. Processed $success_count/$file_count files successfully."
echo "Processed $success_count/$file_count files successfully at $(date)" >> "$LOG_FILE"
echo "Combined RAFT data saved to: $COMBINED_JSONL"

# Optional: Shuffle the combined JSONL for better training
if [ -f "$COMBINED_JSONL" ] && [ -s "$COMBINED_JSONL" ]; then
    echo "Shuffling combined data..."
    TEMP_FILE="$OUTPUT_DIR/temp_combined.jsonl"
    cat "$COMBINED_JSONL" | shuf > "$TEMP_FILE"
    mv "$TEMP_FILE" "$COMBINED_JSONL"
    echo "Combined data shuffled successfully."
fi

echo "All processing complete."
