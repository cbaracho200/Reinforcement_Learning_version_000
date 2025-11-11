"""Domínio de Marketing."""

try:
    from business_rl.domains.marketing.campanha_ads import CampanhaAds
    __all__ = ['CampanhaAds']
except ImportError:
    __all__ = []
