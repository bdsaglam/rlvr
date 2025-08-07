from typing import List

from verifiers.parsers.xml_parser import XMLParser

from verifiers.models import RewardFunc


def make_xml_reward_func(parser: XMLParser) -> RewardFunc:
    """
    Return a reward function that checks for proper XML tag usage.
    
    The returned function evaluates if messages in trajectories properly use 
    the expected XML tags defined in this parser's fields configuration.
    """
    def xml_reward_func(completions, **kwargs) -> List[float]:
        """Reward function that checks for proper XML tag usage in completions."""
        def count_xml(trajectory) -> float:
            # Get all messages from the model
            model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
            if not model_messages:
                return 0.0
            
            # Calculate XML tag usage scores for each message
            xml_scores = []
            for msg in model_messages:
                content = msg['content']
                score = 0
                total_checks = 0
                
                # For each canonical field with its alternatives
                for canonical, alternatives in parser._fields:
                    # Track if at least one alternative was used for this field
                    field_used = False
                    
                    # Check all alternatives for this field
                    for alt in alternatives:
                        # If this alternative is used, check it has proper tags
                        if content.count(f"<{alt}>") > 0 or content.count(f"</{alt}>") > 0:
                            field_used = True
                            score += 1 - abs(content.count(f"<{alt}>") - 1)
                            score += 1 - abs(content.count(f"</{alt}>") - 1)
                            total_checks += 2
                    
                    # If no alternatives for this field were used, we don't add to total_checks
                    # because we're not requiring any specific field to be present
                
                # Calculate normalized score for this message
                if total_checks > 0:
                    xml_scores.append(score / total_checks)
                else:
                    # If no tags used at all, give a zero score
                    xml_scores.append(0.0)
            
            # Return average XML score across all messages
            if not xml_scores:
                return 0.0
            return 0.2 * (sum(xml_scores) / len(xml_scores))  # 0.2 weight as in both rubrics
        
        # Apply the XML check to each completion trajectory
        return [count_xml(c) for c in completions]

    return xml_reward_func

def make_format_reward_func(parser: XMLParser) -> RewardFunc:
    """
    Return a reward function that checks if messages follow the expected format.
    
    The function does not make assumptions about which fields should start/end the message
    or the specific order of fields. It checks that:
    - At least one field from the schema is present in each message
    - Fields have proper content and spacing
    """
    def format_reward_func(completions, **kwargs) -> List[float]:
        """Reward function that checks if each step follows the expected format."""
        def check_format(trajectory):
            # Get assistant messages
            model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
            if not model_messages:
                return 0.0
            
            # Calculate format adherence for each message
            format_scores = []
            for msg in model_messages:
                content = msg['content']
                parsed = parser.parse(content)
                parsed_no_strip = parser.parse(content, strip=False)
                
                # Check if the message has at least one valid field
                has_any_field = False
                fields_with_content = 0
                total_fields = 0
                
                # Keep track of which expected fields are present
                expected_field_count = len(parser._fields)  # Total number of expected field sets
                present_field_sets = set()  # Which field sets have at least one alternative present
                
                # Check proper spacing for fields
                has_correct_spacing = True
                
                for i, (canonical, alternatives) in enumerate(parser._fields):
                    field_set_present = False
                    for alt in alternatives:
                        if val := getattr(parsed, alt, None):
                            has_any_field = True
                            fields_with_content += 1
                            total_fields += 1
                            field_set_present = True
                            
                            # Check if field exists in non-stripped version too (proper spacing)
                            val_no_strip = getattr(parsed_no_strip, alt, None)
                            if val != val_no_strip:
                                has_correct_spacing = False
                        elif content.count(f"<{alt}>") > 0 or content.count(f"</{alt}>") > 0:
                            # Tag exists but content wasn't properly parsed
                            total_fields += 1
                            field_set_present = True
                    
                    # If any alternative from this field set was present, count it
                    if field_set_present:
                        present_field_sets.add(i)
                
                # Calculate format score components
                format_score = 0.0
                
                # Check if any field from the first field set starts the message
                starts_with_any_field = False
                first_field_set = parser._fields[0][1]  # Get alternatives for first field set
                for alt in first_field_set:
                    if content.strip().startswith(f"<{alt}>"):
                        starts_with_any_field = True
                        break
                
                # Check if any field from the last field set ends the message
                ends_with_any_field = False
                last_field_set = parser._fields[-1][1]  # Get alternatives for last field set
                for alt in last_field_set:
                    if content.strip().endswith(f"</{alt}>"):
                        ends_with_any_field = True
                        break
                
                # Weight the score based on different criteria
                if has_any_field:
                    # Calculate the proportion of expected field sets that are present
                    field_set_ratio = len(present_field_sets) / expected_field_count
                    format_score += 0.4 * field_set_ratio
                
                if has_correct_spacing:
                    format_score += 0.2
                
                if starts_with_any_field:
                    format_score += 0.2
                    
                if ends_with_any_field:
                    format_score += 0.2
                
                format_scores.append(format_score)
            
            # Return average format adherence
            if not format_scores:
                return 0.0
            return 0.2 * (sum(format_scores) / len(format_scores))  # 0.2 weight as in both rubrics
        
        # Apply the format check to each completion trajectory
        return [check_format(c) for c in completions]

    return format_reward_func
