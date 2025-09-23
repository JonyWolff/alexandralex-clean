import sys
sys.path.append('/Users/jonywolff/alexandralex-fresh')

from pinecone import Pinecone

pc = Pinecone(api_key='pcsk_6VFPHB_D2TwrH3vHwD4WPYNibbGAM7n8SsMhVHU897gegwzP9WqTf3TBER8EY1csN6X7AV')
index = pc.Index('alexandralex')
stats = index.describe_index_stats()
print(f'✅ Pinecone: {stats.total_vector_count} vetores')
