import './App.css';
import { BrowserRouter, Routes, Route, } from "react-router-dom";
import LoginPage from './pages/Login';
import NLPPage from './pages/NLP';

function App() {
  return (
    <div className="min-h-full h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<LoginPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/nlp" element={<NLPPage />} />
          </Routes>
        </BrowserRouter>
      </div>
    </div>
  );
}

export default App;