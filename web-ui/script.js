const dropZone1 = document.getElementById('drop-zone-1');
const dropZone2 = document.getElementById('drop-zone-2');
const fileInput1 = document.getElementById('file-input-1');
const fileInput2 = document.getElementById('file-input-2');
const processBtn = document.getElementById('process-btn');
const baselineInput = document.getElementById('baseline');
const resultContainer = document.getElementById('result-container');
const resultImg = document.getElementById('result-img');
const loader = document.getElementById('loader');

const resWidth = document.getElementById('res-width');
const resHeight = document.getElementById('res-height');
const resDepth = document.getElementById('res-depth');

let file1 = null;
let file2 = null;

// Handle selection for Photo A
dropZone1.addEventListener('click', () => fileInput1.click());
fileInput1.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        file1 = e.target.files[0];
        dropZone1.querySelector('h3').textContent = file1.name;
        dropZone1.style.borderColor = 'var(--primary)';
    }
});

// Handle selection for Photo B
dropZone2.addEventListener('click', () => fileInput2.click());
fileInput2.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        file2 = e.target.files[0];
        dropZone2.querySelector('h3').textContent = file2.name;
        dropZone2.style.borderColor = 'var(--primary)';
    }
});

// Process button handler
processBtn.addEventListener('click', async () => {
    if (!file1 || !file2) {
        alert('Please upload both Photo A and Photo B.');
        return;
    }

    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);
    formData.append('baseline', baselineInput.value);

    // Show loading state
    resultContainer.classList.remove('hidden');
    resultImg.style.opacity = '0.3';
    loader.classList.remove('hidden');
    processBtn.disabled = true;
    processBtn.textContent = 'Processing 3D Data...';

    try {
        const response = await fetch('http://localhost:8000/estimate', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            const imageUrl = `http://localhost:8000${data.output_url}`;
            resultImg.src = imageUrl;
            resultImg.style.opacity = '1';
            
            // Update stats
            resWidth.textContent = data.dimensions.width.toFixed(1) + 'mm';
            resHeight.textContent = data.dimensions.height.toFixed(1) + 'mm';
            resDepth.textContent = data.dimensions.depth.toFixed(1) + 'mm';
            
        } else {
            alert('Error: ' + data.error);
            resultContainer.classList.add('hidden');
        }
    } catch (err) {
        console.error(err);
        alert('Could not connect to the backend server.');
        resultContainer.classList.add('hidden');
    } finally {
        loader.classList.add('hidden');
        processBtn.disabled = false;
        processBtn.textContent = 'Calculate 3D Dimensions';
    }
});
