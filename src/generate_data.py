"""
Generate synthetic GC/HPLC analysis data for petroleum products.
Simulates Gas Chromatography outputs for predicting octane number.
"""

import pandas as pd
import numpy as np
import os


def generate_gc_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic Gas Chromatography data for gasoline samples.
    
    Features are component percentages detected by GC:
    - Paraffins (saturated hydrocarbons)
    - Olefins (unsaturated hydrocarbons)
    - Naphthenes (cyclic hydrocarbons)
    - Aromatics (benzene ring compounds)
    - Oxygenates (MTBE, Ethanol)
    
    Target: Research Octane Number (RON) - typically 87-98 for gasoline
    """
    np.random.seed(42)
    
    # Generate component percentages (must sum to ~100%)
    # Different fuel grades have different compositions
    
    # Create base compositions for different grades
    data = []
    
    for i in range(n_samples):
        # Randomly select a grade profile
        grade = np.random.choice(['regular', 'mid', 'premium'], p=[0.4, 0.3, 0.3])
        
        if grade == 'regular':
            # Regular: Higher paraffins, lower aromatics
            paraffins = np.random.normal(55, 5)
            olefins = np.random.normal(12, 3)
            naphthenes = np.random.normal(8, 2)
            aromatics = np.random.normal(20, 4)
            oxygenates = np.random.normal(5, 2)
            base_ron = 87
        elif grade == 'mid':
            # Mid-grade: Balanced composition
            paraffins = np.random.normal(48, 5)
            olefins = np.random.normal(10, 3)
            naphthenes = np.random.normal(10, 2)
            aromatics = np.random.normal(25, 4)
            oxygenates = np.random.normal(7, 2)
            base_ron = 89
        else:
            # Premium: Higher aromatics and oxygenates
            paraffins = np.random.normal(40, 5)
            olefins = np.random.normal(8, 3)
            naphthenes = np.random.normal(12, 2)
            aromatics = np.random.normal(32, 4)
            oxygenates = np.random.normal(8, 2)
            base_ron = 93
        
        # Normalize to sum to 100%
        total = paraffins + olefins + naphthenes + aromatics + oxygenates
        paraffins = (paraffins / total) * 100
        olefins = (olefins / total) * 100
        naphthenes = (naphthenes / total) * 100
        aromatics = (aromatics / total) * 100
        oxygenates = (oxygenates / total) * 100
        
        # Calculate RON based on composition
        # Aromatics and oxygenates increase octane
        # Paraffins generally decrease it
        ron = (base_ron 
               + 0.15 * (aromatics - 25)  # Aromatics boost
               + 0.3 * (oxygenates - 5)   # Oxygenates boost
               - 0.1 * (paraffins - 45)   # Paraffins penalty
               + 0.05 * naphthenes        # Slight naphthene boost
               + np.random.normal(0, 1))  # Random noise
        
        # Additional GC metrics
        density = np.random.normal(0.74, 0.02)  # g/mL
        rvp = np.random.normal(9.0, 1.5)  # Reid Vapor Pressure (psi)
        distillation_t50 = np.random.normal(100, 10)  # °C at 50% distilled
        distillation_t90 = np.random.normal(160, 15)  # °C at 90% distilled
        sulfur_ppm = np.random.exponential(15)  # Parts per million
        benzene_pct = np.clip(np.random.normal(0.8, 0.3), 0.1, 2.0)  # %
        
        data.append({
            'sample_id': f'GC-{i+1:04d}',
            'paraffins_pct': round(paraffins, 2),
            'olefins_pct': round(olefins, 2),
            'naphthenes_pct': round(naphthenes, 2),
            'aromatics_pct': round(aromatics, 2),
            'oxygenates_pct': round(oxygenates, 2),
            'density_gml': round(density, 4),
            'rvp_psi': round(rvp, 2),
            'distillation_t50_c': round(distillation_t50, 1),
            'distillation_t90_c': round(distillation_t90, 1),
            'sulfur_ppm': round(sulfur_ppm, 1),
            'benzene_pct': round(benzene_pct, 2),
            'ron': round(np.clip(ron, 84, 100), 1),  # Research Octane Number
            'grade': grade
        })
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['aromatic_paraffin_ratio'] = round(df['aromatics_pct'] / df['paraffins_pct'], 3)
    df['distillation_range'] = df['distillation_t90_c'] - df['distillation_t50_c']
    
    return df


def main():
    print("Generating synthetic GC/HPLC analysis data...")
    
    df = generate_gc_data(n_samples=1000)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'gc_analysis.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} samples")
    print(f"RON range: {df['ron'].min():.1f} - {df['ron'].max():.1f}")
    print(f"Saved to: {output_path}")
    
    # Summary
    print("\nGrade Distribution:")
    print(df['grade'].value_counts())
    
    print("\nFeature Statistics:")
    print(df[['paraffins_pct', 'aromatics_pct', 'oxygenates_pct', 'ron']].describe())


if __name__ == "__main__":
    main()
