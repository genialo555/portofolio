"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Revenue {
  sponsoredPosts: number;
  affiliateMarketing: number;
  productSales: number;
  coaching: number;
}

interface KPIFinancier {
  revenue: Revenue;
}

interface FinancialSectionProps {
  kpis: KPIFinancier;
}

const mockKPIs: KPIFinancier = {
  revenue: {
    sponsoredPosts: 5000,
    affiliateMarketing: 3000,
    productSales: 7500,
    coaching: 4500
  }
};

export function FinancialSection({ kpis = mockKPIs }: FinancialSectionProps) {
  const totalRevenue = Object.values(kpis.revenue).reduce((a, b) => a + b, 0);

  const getRevenuePercentage = (value: number) => {
    return (value / totalRevenue) * 100;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('fr-FR', {
      style: 'currency',
      currency: 'EUR',
      maximumFractionDigits: 0
    }).format(value);
  };

  return (
    <Card className="bg-white/5 backdrop-blur-sm border-gray-800">
      <CardHeader className="flex justify-between items-center">
        <CardTitle className="text-lg font-semibold text-white">Aperçu Financier</CardTitle>
        <div className="text-sm text-gray-400 font-medium">Ce mois-ci</div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Sponsored Posts */}
        <div className="space-y-4">
          <div className="flex justify-between items-baseline">
            <h4 className="text-gray-300">Posts Sponsorisés</h4>
            <span className="text-2xl font-bold text-white">
              {formatCurrency(kpis.revenue.sponsoredPosts)}
            </span>
          </div>
          <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full transition-all duration-500"
              style={{ width: `${getRevenuePercentage(kpis.revenue.sponsoredPosts)}%` }}
            />
          </div>
        </div>

        {/* Affiliate Marketing */}
        <div className="space-y-4">
          <div className="flex justify-between items-baseline">
            <h4 className="text-gray-300">Marketing d'Affiliation</h4>
            <span className="text-2xl font-bold text-white">
              {formatCurrency(kpis.revenue.affiliateMarketing)}
            </span>
          </div>
          <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full transition-all duration-500"
              style={{ width: `${getRevenuePercentage(kpis.revenue.affiliateMarketing)}%` }}
            />
          </div>
        </div>

        {/* Product Sales */}
        <div className="space-y-4">
          <div className="flex justify-between items-baseline">
            <h4 className="text-gray-300">Ventes de Produits</h4>
            <span className="text-2xl font-bold text-white">
              {formatCurrency(kpis.revenue.productSales)}
            </span>
          </div>
          <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full transition-all duration-500"
              style={{ width: `${getRevenuePercentage(kpis.revenue.productSales)}%` }}
            />
          </div>
        </div>

        {/* Coaching */}
        <div className="space-y-4">
          <div className="flex justify-between items-baseline">
            <h4 className="text-gray-300">Coaching</h4>
            <span className="text-2xl font-bold text-white">
              {formatCurrency(kpis.revenue.coaching)}
            </span>
          </div>
          <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full transition-all duration-500"
              style={{ width: `${getRevenuePercentage(kpis.revenue.coaching)}%` }}
            />
          </div>
        </div>

        {/* Total Revenue */}
        <div className="pt-6 border-t border-gray-800">
          <div className="flex justify-between items-baseline">
            <span className="text-gray-300">Revenu Total</span>
            <span className="text-3xl font-bold text-white">
              {formatCurrency(totalRevenue)}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 