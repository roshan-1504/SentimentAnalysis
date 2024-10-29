import { useState } from 'react';
import { loginFields } from "../constants/constants";
import Input from "./Input";

const fields = loginFields;

let fieldsState = {};
fields.forEach(field => fieldsState[field.id] = '');

export default function Login() {
    const [loginState, setLoginState] = useState(fieldsState);

    const [flashMessage, setFlashMessage] = useState({
        text: "",
        success: false,
        failure: false,
    });

    const handleFlashMessage = (text, success) => {
        setFlashMessage({ text, success, failure: !success });
        setTimeout(() => setFlashMessage({ text: "", success: false, failure: false }), 2000);
    };

    const handleChange = (e) => {
        setLoginState({ ...loginState, [e.target.id]: e.target.value });
    }

    const handleSubmit = (e) => {
        e.preventDefault();

        // Check if required fields are entered
        if (!loginState.username) {
            handleFlashMessage("Please enter an username", false, 2000);
            return;
        } else {
            authenticateUser();
        }
    }

    const authenticateUser = () => {
        const usernameInput = loginState.username;

        fetch("/login", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: usernameInput
            }),
        })
            .then(async (response) => {
                const data = await response.json();

                if (response.ok) {
                    if (data.message === 'Login successful') {
                        handleFlashMessage('Login successful!', true);
                        window.location.href = '/nlp';
                    }
                } else if (response.status === 401) {
                    handleFlashMessage('Invalid username. Please try again.', false);
                } else if (response.status === 500) {
                    handleFlashMessage('Internal server error. Please try again later.', false);
                } else {
                    handleFlashMessage('An error occurred. Please try again.', false);
                }
            })
            .catch((error) => {
                console.error("Error logging in:", error);
                handleFlashMessage('Network error. Please check your connection.', false);
            })
            .finally(() => {
                resetForms();
            });
    };

    const resetForms = () => {
        setLoginState(fieldsState);
    };


    return (
        <div>

            {flashMessage.failure && (
                <div id="failFlashMsg">
                    {flashMessage.text}
                </div>
            )}

            <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
                <div className="-space-y-px">
                    {fields.map(field =>
                        <Input
                            key={field.id}
                            handleChange={handleChange}
                            value={loginState[field.id]}
                            labelText={field.labelText}
                            labelFor={field.labelFor}
                            id={field.id}
                            name={field.name}
                            type={field.type}
                            isRequired={field.isRequired}
                            placeholder={field.placeholder}
                            maxLength={field.maxLength}
                        />
                    )}
                </div>

                <button className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 mt-10" onClick={handleSubmit}>Login</button>
            </form>

        </div>
    )
}
