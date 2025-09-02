
import React from 'react';
import type { UsageChartData } from '../types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface UsageChartComponentProps {
    chartData: UsageChartData;
}

const COLORS = ['#38bdf8', '#fbbf24', '#a78bfa', '#f87171'];

export const UsageChart: React.FC<UsageChartComponentProps> = ({ chartData }) => {
    if (!chartData || !chartData.data || chartData.data.length === 0) {
        return <p>No usage data available for this chart.</p>;
    }

    const years = Array.from(new Set(chartData.data.flatMap(d => d.usage.map(u => u.year)))).sort();

    const processedData = chartData.data.map(item => {
        const dataPoint: { [key: string]: string | number } = { month: item.month };
        item.usage.forEach(u => {
            dataPoint[u.year] = u.value;
        });
        return dataPoint;
    });

    return (
        <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart
                    data={processedData}
                    margin={{
                        top: 5,
                        right: 20,
                        left: -10,
                        bottom: 5,
                    }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(128, 128, 128, 0.2)" />
                    <XAxis dataKey="month" tick={{ fill: 'rgb(100 116 139)', fontSize: 12 }} />
                    <YAxis
                        tick={{ fill: 'rgb(100 116 139)', fontSize: 12 }}
                        label={{ value: chartData.unit, angle: -90, position: 'insideLeft', fill: 'rgb(100 116 139)', style: { textAnchor: 'middle' } }}
                    />
                    <Tooltip
                        cursor={{ fill: 'rgba(100, 116, 139, 0.1)' }}
                        contentStyle={{
                            backgroundColor: 'rgb(15 23 42)',
                            borderColor: 'rgb(51 65 85)',
                            color: 'rgb(226 232 240)',
                            borderRadius: '0.5rem'
                        }}
                    />
                    <Legend wrapperStyle={{fontSize: "14px"}}/>
                    {years.map((year, index) => (
                        <Bar key={year} dataKey={year} fill={COLORS[index % COLORS.length]} />
                    ))}
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};
