"""Domínio de Real Estate."""

try:
    from business_rl.domains.real_estate.compra_terreno import CompraTerreno
    __all__ = ['CompraTerreno']
except ImportError:
    __all__ = []
