document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('result-section');
    const previewImg = document.getElementById('preview-img');
    const predClass = document.getElementById('pred-class');
    const predConf = document.getElementById('pred-conf');
    const resetBtn = document.getElementById('reset-btn');

    // Handle click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileUpload(fileInput.files[0]);
        }
    });

    // Handle file selection
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFileUpload(fileInput.files[0]);
        }
    });

    // Reset UI
    resetBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        fileInput.value = '';
    });

    function handleFileUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Show loader
        uploadArea.classList.add('hidden');
        loader.classList.remove('hidden');

        // Prepare form data
        const formData = new FormData();
        formData.append('file', file);

        // Send to backend
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loader.classList.add('hidden');
            
            if (data.error) {
                alert('Error: ' + data.error);
                uploadArea.classList.remove('hidden');
            } else {
                // Show results
                previewImg.src = data.image_url;
                predClass.textContent = data.predicted_class;
                predConf.textContent = data.confidence;
                
                // Color text based on prediction for extra flair (optional)
                if (data.predicted_class.toLowerCase() === 'venomous') {
                    predClass.style.color = '#ff4c4c'; // Red-ish for warning
                } else {
                    predClass.style.color = 'var(--gold)'; // Keep gold
                }

                resultSection.classList.remove('hidden');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during prediction.');
            loader.classList.add('hidden');
            uploadArea.classList.remove('hidden');
        });
    }
});
