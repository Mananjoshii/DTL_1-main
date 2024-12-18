const express = require('express');
const multer = require('multer');
const FormData = require('form-data'); // Import the form-data package
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(express.static(path.join(__dirname, 'public')));

app.post('/upload', upload.single('file'), async (req, res) => {
    try {
        const file = req.file;
        console.log("File received:", file);

        // Create FormData instance
        const formData = new FormData();
        formData.append('file', fs.createReadStream(file.path)); // Attach the file

        // Send the form data to the FastAPI backend
        const response = await axios.post('http://localhost:8000/predict/', formData, {
            headers: formData.getHeaders(), // Set appropriate headers
        });

        console.log("Backend response:", response.data); // Log backend response
        res.json(response.data);
    } catch (error) {
        console.error("Error predicting file:", error); // Log error details
        res.status(500).send("Error in prediction.");
    }
});

app.listen(3000, () => {
    console.log('Frontend server running at http://localhost:3000');
});
