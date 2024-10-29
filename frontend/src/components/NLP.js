import { useState, useEffect } from 'react';
import { nlpFields } from "../constants/constants";
import Input from "./Input";

const fields = nlpFields;

let fieldsState = {};
fields.forEach(field => fieldsState[field.id] = '');

export default function NLP() {
    const [NLPState, setNLPState] = useState(fieldsState);

    const [flashMessage, setFlashMessage] = useState({
        text: "",
        success: false,
        failure: false,
    });

    const [selectedModels, setSelectedModels] = useState([]);
    const [modelOptions, setModelOptions] = useState([]);
    const [divNumber, setDivNumber] = useState(1);
    const [predictions, setPredictions] = useState([]);

    // Fetch model options from backend
    useEffect(() => {
        fetch("/models")
            .then(response => response.json())
            .then(data => setModelOptions(data))
            .catch(error => console.error('Error fetching model options:', error));
    }, []);

    const handleFlashMessage = (text, success, time) => {
        setFlashMessage({ text, success, failure: !success });
        setTimeout(() => setFlashMessage({ text: "", success: false, failure: false }), time);
    };

    const handleChange = (e) => {
        setNLPState({ ...NLPState, [e.target.id]: e.target.value });
    }

    const handleModelChange = (e) => {
        const model = e.target.value;
        if (e.target.checked) {
            // Add the model to the selected list
            setSelectedModels([...selectedModels, model]);
        } else {
            // Remove the model from the selected list
            setSelectedModels(selectedModels.filter(m => m !== model));
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();

        if (!NLPState.statement) {
            handleFlashMessage("Please enter a statement", false, 2000);
            return;
        } else if (selectedModels.length === 0) {
            handleFlashMessage("Please select at least one model", false, 2000);
            return;
        } else {
            handlePredict();
            setDivNumber(2);
        }
    };

    const handlePredict = () => {
        const statementInput = NLPState.statement;

        fetch("/predict", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                statement: statementInput,
                models: selectedModels
            }),
        })
            .then(async (response) => {
                const data = await response.json();
                console.log(data.predictions);
                setPredictions(data.predictions); // Save predictions for display

                if (response.ok) {
                    handleFlashMessage('Prediction successful!', true, 2000);
                } else {
                    handleFlashMessage('An error occurred. Please try again.', false, 2000);
                }
            })
            .catch((error) => {
                console.error("Error during prediction:", error);
                handleFlashMessage('Network error. Please check connection.', false, 2000);
            })
    };

    const handleBack = () => {
        setDivNumber(1);
        setNLPState(fieldsState);
        setSelectedModels([]);
        setPredictions([]);
    };


    return (
        <div>
            {flashMessage.success && (
                <div id="successFlashMsg" style={{ marginTop: '15px' }}>
                    {flashMessage.text}
                </div>
            )}

            {flashMessage.failure && (
                <div id="failFlashMsg" style={{ marginTop: '15px' }}>
                    {flashMessage.text}
                </div>
            )}

            <div>
                <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
                    <div className="-space-y-px">
                        {nlpFields.map(field =>
                            <Input
                                key={field.id}
                                handleChange={handleChange}
                                value={NLPState[field.id]}
                                labelText={field.labelText}
                                labelFor={field.labelFor}
                                id={field.id}
                                name={field.name}
                                type={field.type}
                                isRequired={field.isRequired}
                                placeholder={field.placeholder}
                                maxLength={field.maxLength}
                                disabled={divNumber === 2}
                            />
                        )}
                    </div>

                    {divNumber === 1 && (
                        <div>
                            <div className="mt-4">
                                <p style={{
                                    fontFamily: 'Arial, sans-serif',
                                    fontSize: '16px',
                                    color: '#4A4A4A',
                                    fontWeight: '500'
                                }}>
                                    Select models to use:
                                </p>
                                {modelOptions.map((model, index) => (
                                    <div key={index}>
                                        <input
                                            type="checkbox"
                                            id={`model-${index}`}
                                            value={model.id}
                                            onChange={handleModelChange}
                                            checked={selectedModels.includes(model.id)}
                                        />
                                        <label
                                            htmlFor={`model-${index}`}
                                            className="ml-2"
                                            style={{
                                                fontFamily: 'Arial, sans-serif',
                                                fontSize: '16px',
                                                color: '#4A4A4A',
                                                fontWeight: '500'
                                            }}>
                                            {model.name}
                                        </label>
                                    </div>
                                ))}
                            </div>

                            <button
                                className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 mt-10"
                                onClick={handleSubmit}
                            >
                                Predict
                            </button>
                        </div>
                    )}

                    {divNumber === 2 && (
                        <div className="mt-4">
                            <h2 style={{
                                fontFamily: 'Arial, sans-serif',
                                fontSize: '20px',
                                color: '#4A4A4A',
                                fontWeight: '600',
                                marginBottom: '10px'
                            }}>
                                Prediction Results:
                            </h2>

                            {/* Table for displaying predictions */}
                            <table style={{
                                width: '100%',
                                borderCollapse: 'collapse',
                                fontFamily: 'Arial, sans-serif',
                                fontSize: '16px',
                                color: '#4A4A4A'
                            }}>
                                <thead>
                                    <tr style={{ backgroundColor: '#f2f2f2' }}>
                                        <th style={{ border: '1px solid #ddd', padding: '8px', fontWeight: '600' }}>Model</th>
                                        <th style={{ border: '1px solid #ddd', padding: '8px', fontWeight: '600' }}>Prediction</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {predictions.map((prediction, index) => {
                                        // Determine row color based on prediction type
                                        let rowColor = '#fff'; // Default to white
                                        if (prediction.prediction.toLowerCase() === 'negative') {
                                            rowColor = '#f8d7da'; // Light red for negative
                                        } else if (prediction.prediction.toLowerCase() === 'positive') {
                                            rowColor = '#d4edda'; // Light green for positive
                                        } else if (prediction.prediction.toLowerCase() === 'neutral') {
                                            rowColor = '#ffffff'; // White for neutral
                                        }

                                        // Capitalize the first letter of the prediction
                                        const capitalizedPrediction = prediction.prediction.charAt(0).toUpperCase() + prediction.prediction.slice(1);

                                        return (
                                            <tr key={index} style={{ backgroundColor: rowColor, borderBottom: '1px solid #ddd' }}>
                                                <td style={{ border: '1px solid #ddd', padding: '8px' }}>{prediction.model}</td>
                                                <td style={{ border: '1px solid #ddd', padding: '8px' }}>{capitalizedPrediction}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>

                            {/* Container for back link */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px' }}>
                                <div></div> {/* Empty div for spacing */}
                                <a
                                    href="#"
                                    style={{
                                        fontSize: '14px',
                                        color: '#007BFF',
                                        textDecoration: 'underline',
                                        cursor: 'pointer',
                                    }}
                                    onClick={handleBack} // Use handleBack function
                                >
                                    Back
                                </a>
                            </div>
                        </div>
                    )}


                </form>
            </div>

        </div>
    )
}
