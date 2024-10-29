const loginFields = [
    {
        labelText: "Username",
        labelFor: "username",
        id: "username",
        name: "username",
        type: "text",
        autoComplete: "username",
        isRequired: true,
        placeholder: "Username",
        maxLength: 30
    }
]

const nlpFields = [
    {
        labelText: 'Statement',
        labelFor: 'statement',
        id: 'statement',
        name: 'statement',
        type: 'text',
        isRequired: true,
        placeholder: 'Enter a statement',
        maxLength: 50
    }
];


export { loginFields, nlpFields };