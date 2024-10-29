const express = require('express');
const session = require('express-session');
const crypto = require('crypto');
const path = require('path');
const axios = require('axios');
const xss = require('xss');
const yaml = require('js-yaml');
const fs = require('fs');
const app = express();

app.use(express.json());

const buildPath = path.join(__dirname, '..', 'build');
app.use(express.static(buildPath));

app.use(
    session({
        secret: crypto.randomBytes(32).toString('hex'),
        resave: false,
        saveUninitialized: false,
    })
);

app.post('/login', async (req, res) => {
    const username = req.body.username;

    try {
        const sanitizedUsername = xss(username);
        if (sanitizedUsername === "NLP") {
            req.session.username = sanitizedUsername;
            res.status(200).send({ message: 'Login successful' });
        } else {
            res.status(401).json({ message: 'Unauthorized: Invalid username' });
        }
    } catch (error) {
        console.error('Error:', error.message);
        res.status(500).json({ message: 'Internal server error' });
    }
});

app.get('/', (req, res) => {
    res.sendFile(path.join(buildPath, 'index.html'));
});

app.get('/nlp', (req, res) => {
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    const username = req.session.username;
    if (!username) {
        return res.redirect("/");
    }
    res.sendFile(path.join(buildPath, 'index.html'));
});

// Load model options from config.yaml
let modelOptions;
try {
    const config = yaml.load(fs.readFileSync(path.join(__dirname, 'config.yaml'), 'utf8'));
    modelOptions = config.models;
} catch (e) {
    console.error('Failed to load config.yaml:', e.message);
}

// Serve model options to the frontend
app.get('/models', (req, res) => {
    res.json(modelOptions);
});

app.post('/predict', async (req, res) => {
    const { statement, models } = req.body;

    console.log('\nReceived statement:', statement);
    console.log('Selected models:', models);

    try {
        // Prepare an array of axios requests for the selected models
        const predictionPromises = models.map(modelId => {
            // Find the model configuration by its id
            const modelConfig = modelOptions.find(model => model.id === modelId);
            if (modelConfig && modelConfig.serverUrl) {
                // Send prediction request to the server URL
                return axios.post(modelConfig.serverUrl, { statement })
                    .then(response => {
                        // Log the response received from the model server
                        console.log(`Response received from ${modelConfig.name}:`, response.data.prediction);
                        return {
                            model: modelConfig.name,
                            prediction: response.data.prediction, // Only return the prediction
                        };
                    })
                    .catch(error => {
                        console.error(`Error fetching prediction from ${modelConfig.name}:`, error.message);
                        return { model: modelConfig.name, error: 'Failed to get prediction' };
                    });
            } else {
                return Promise.resolve({ model: modelId, error: 'Server URL not configured' });
            }
        });

        // Await all promises to resolve
        const predictions = await Promise.all(predictionPromises);

        // Send the aggregated predictions back to the client
        res.status(200).json({
            message: 'Predictions received successfully',
            predictions, // This will contain an array of model and prediction pairs
        });
    } catch (error) {
        console.error('Error processing prediction requests:', error.message);
        res.status(500).json({ message: 'Internal server error' });
    }
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
