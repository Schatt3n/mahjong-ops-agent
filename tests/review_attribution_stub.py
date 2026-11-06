"""受控审查夹具的固定归属响应，只验证协议接线，不执行语义分类。"""
def fixed_customer_email_attribution(case):
    """只为预声明正例生成精确来源片段；Agent借用同句的负例不获得归属证明。"""
    if case['case_id'] != 'review_customer_email_attribution':
        return []
    claim = '你已经发了预算邮件'
    source = '我已经发了预算邮件'
    return [{'reviewed_text': case['text'], 'claim_span': [0, len(claim)], 'actor': 'current_customer',
        'actor_span': [0, 1], 'source_ref': 'current_message', 'source_quote': source,
        'source_claim_span': [0, len(source)], 'source_actor_span': [0, 1]}]
