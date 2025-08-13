import React, { useState } from "react";

const MainPage: React.FC = () => {
    const [symbol, setSymbol] = useState("");
    const [prediction, setPrediction] = useState<number | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handlePredict = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setPrediction(null);

        try {
            // Placeholder for API call
            // Replace with your prediction API endpoint
            const response = await fetch(`/api/predict?symbol=${symbol}`);
            if (!response.ok) throw new Error("Failed to fetch prediction");
            const data = await response.json();
            setPrediction(data.predictedPrice);
        } catch (err: any) {
            setError(err.message || "An error occurred");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ maxWidth: 500, margin: "40px auto", padding: 24, border: "1px solid #eee", borderRadius: 8 }}>
            <h1>Stock Price Prediction</h1>
            <form onSubmit={handlePredict} style={{ marginBottom: 24 }}>
                <label>
                    Stock Symbol:
                    <input
                        type="text"
                        value={symbol}
                        onChange={e => setSymbol(e.target.value.toUpperCase())}
                        placeholder="e.g. AAPL"
                        style={{ marginLeft: 8, padding: 4 }}
                        required
                    />
                </label>
                <button type="submit" style={{ marginLeft: 12, padding: "4px 16px" }} disabled={loading}>
                    {loading ? "Predicting..." : "Predict"}
                </button>
            </form>
            {error && <div style={{ color: "red" }}>{error}</div>}
            {prediction !== null && (
                <div>
                    <h2>Predicted Price</h2>
                    <p>
                        The predicted price for <strong>{symbol}</strong> is <strong>${prediction.toFixed(2)}</strong>
                    </p>
                </div>
            )}
        </div>
    );
};

export default MainPage;