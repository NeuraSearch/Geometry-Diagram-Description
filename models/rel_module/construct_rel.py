# coding:utf-8

import torch
import torch.nn as nn

class GeotoGeo(nn.Module):
        
    def __init__(self, geo_embed_size, geo_rel_size):
        super(GeotoGeo, self).__init__()
        
        self.combine_point_line_layer = nn.Sequential(
            nn.Linear(2 * geo_embed_size, geo_rel_size),
            nn.ReLU(),
            nn.LayerNorm(geo_rel_size),
            nn.Linear(geo_rel_size, 3)
        )

        self.combine_point_circle_layer = nn.Sequential(
            nn.Linear(2 * geo_embed_size, geo_rel_size),
            nn.ReLU(),
            nn.LayerNorm(geo_rel_size),
            nn.Linear(geo_rel_size, 3)
        )
        
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nllloss = nn.NLLLoss()
        
    def forward(self, all_geo_info, targets_geo=None):
        
        if self.training:
            pl_loss, pc_loss = self._forward_train(all_geo_info, targets_geo)
            
            losses = {
                "pl_loss": pl_loss,
                "pc_loss": pc_loss,
            }
            return None, losses
        
        else:
            pl_rels, pc_rels = self._forward_test(all_geo_info)
            
            geo_rels_predictions = []   # len==bsz
            for pl, pc in zip(pl_rels, pc_rels):
                geo_rels_predictions.append(
                    {
                        "pl_rels": pl,
                        "pc_rels": pc,
                    }
                )
            
            return geo_rels_predictions, None
    
    def _forward_train(self, all_geo_info, targets_geo):
        
        pl_loss = all_geo_info[0]["points"].new([0]).zero_().squeeze()     # tensor(0.)
        pc_loss = all_geo_info[0]["points"].new([0]).zero_().squeeze()     # tensor(0.)
        for geo_info, target in zip(all_geo_info, targets_geo):
            pl_rel_logits_per_data, pc_rel_logits_per_data = self.construct_geo_to_geo_per_data(geo_info)
            
            if pl_rel_logits_per_data != None:
                pl_loss_per_data = self.cal_geo_rel_loss(pl_rel_logits_per_data, target["pl_rels"])
                pl_loss += pl_loss_per_data
            
            if pc_rel_logits_per_data != None:
                pc_loss_per_data = self.cal_geo_rel_loss(pc_rel_logits_per_data, target["pc_rels"])
                pc_loss += pc_loss_per_data
        
        return pl_loss / len(all_geo_info), pc_loss / len(all_geo_info)

    def _forward_test(self, all_geo_info):
        """
        Returns:
            pl_rels List[Tensor(P, L)]: relation between points and lines.
            pc_rels List[Tensor(P, C)]: relation between points and circle.
        """
        
        pl_rels = []
        pc_rels = []
        for geo_info in all_geo_info:
            pl_rel_logits_per_data, pc_rel_logits_per_data = self.construct_geo_to_geo_per_data(geo_info)
            
            if pl_rel_logits_per_data != None:
                pl_rel_per_data = torch.argmax(pl_rel_logits_per_data, dim=-1)      # [p, l]
                pl_rels.append(pl_rel_per_data)
            else:
                pl_rels.append(None)
            
            if pc_rel_logits_per_data != None:
                pc_rel_per_data = torch.argmax(pc_rel_logits_per_data, dim=-1)      # [p, c]
                pc_rels.append(pc_rel_per_data)
            else:
                pc_rels.append(None)
        
        return pl_rels, pc_rels
            
    def construct_geo_to_geo_per_data(self, geo_info):
        """
            Args:
                geo_info: {"points": Tensor(P, cfg.sym_embed_size), "lines": Tensor(L, cfg.sym_embed_size), "circles": Tensor(C, cfg.sym_embed_size)}
            Returns:
                points_lines_rel: Tensor(p, l, 3)
                point_circles_rel: Tensor(p, c, 3)
        """
        
        points = geo_info["points"]     # [p, h]
        lines = geo_info["lines"]       # [l, h]
        circles = geo_info["circles"]   # [c, h]

        points_lines_rel = None
        if len(points) != 0 and len(lines) != 0:
            # combine point and line
            points_expand = torch.unsqueeze(points, 1).repeat(1, lines.size(0), 1)              # [p, l, h]
            lines_expand = torch.unsqueeze(lines, 0).repeat(points.size(0), 1, 1)               # [p, l, h]
            points_lines = torch.cat((points_expand, lines_expand), dim=2)                      # [p, l, 2h]
            points_lines_rel = self.combine_point_line_layer(points_lines) # [p, l, 3]

        point_circles_rel = None
        if len(points) != 0 and len(circles) != 0:
            # combine point and circle
            points_expand = torch.unsqueeze(points, 1).repeat(1, circles.size(0), 1)             # [p, c, h]
            circles_expand = torch.unsqueeze(circles, 0).repeat(points.size(0), 1, 1)            # [p, c, h]
            points_circles = torch.cat((points_expand, circles_expand), dim=2)                   # [p, c, 2h]
            point_circles_rel = self.combine_point_circle_layer(points_circles)  # [p, c, 3]
        
        return points_lines_rel, point_circles_rel

    def cal_geo_rel_loss(self, rel_per_data, target):
        """
        Args:
            rel_per_data (Tensor(p, ?, 3)): logits
            target (Tensor(p, ?)): !!! remember to convert to tensor and send to GPU in data_loader.
        """
        
        rel_per_data_flatten = torch.reshape(rel_per_data, (-1, 3))     # [p*?, 3]
        target_flatten = torch.reshape(target, (-1,)).to(torch.long)   # [p*?]
        
        rel_per_data_log_prob = self.log_softmax(rel_per_data_flatten)
    
        rel_loss = self.nllloss(input=rel_per_data_log_prob, target=target_flatten) # [1]
        
        return rel_loss



class SymtoGeo(nn.Module):

    def __init__(self, sym_embed_size, sym_rel_size, weak_mask):
        super(SymtoGeo, self).__init__()

        self.weak_mask = weak_mask
        
        self.LPL_layer = nn.Sequential(
            nn.Linear(sym_embed_size, sym_rel_size),
            nn.ReLU(),
            nn.LayerNorm(sym_embed_size),
        )
        
        self.PLP_layer = nn.Sequential(
            nn.Linear(sym_embed_size, sym_rel_size),
            nn.ReLU(),
            nn.LayerNorm(sym_embed_size),
        )
        
        self.PCP_layer = nn.Sequential(
            nn.Linear(sym_embed_size, sym_rel_size),
            nn.ReLU(),
            nn.LayerNorm(sym_embed_size),
        )
        
        self.geo_matrix_layer = nn.Sequential(
            nn.Linear(sym_rel_size, sym_rel_size),
            nn.ReLU(),
            nn.LayerNorm(sym_embed_size),
            nn.Linear(sym_rel_size, sym_rel_size),
            nn.ReLU(),
            nn.LayerNorm(sym_embed_size),
        )
        
        self.text_symbol_to_geo_layer = nn.Sequential(
            nn.Linear(sym_rel_size, sym_rel_size),
            nn.ReLU(),
            nn.LayerNorm(sym_rel_size),
            nn.Linear(sym_rel_size, 1),        
        )
        
        self.head_symbol_to_geo_layer = nn.Sequential(
            nn.Linear(sym_rel_size, sym_rel_size),
            nn.ReLU(),
            nn.LayerNorm(sym_rel_size),
            nn.Linear(sym_rel_size, 1),
        )
        
        self.angle_symbol_to_geo_layer = nn.Sequential(
            nn.Linear(sym_rel_size, sym_rel_size),
            nn.ReLU(),
            nn.LayerNorm(sym_rel_size),
            nn.Linear(sym_rel_size, 1),       
        )
    
        self.bar_symbol_to_geo_layer = nn.Sequential(
            nn.Linear(sym_rel_size, sym_rel_size),
            nn.ReLU(),
            nn.LayerNorm(sym_rel_size),
            nn.Linear(sym_rel_size, 1),
        )
    
        self.parallel_symbol_to_geo_layer = nn.Sequential(
            nn.Linear(sym_rel_size, sym_rel_size),
            nn.ReLU(),
            nn.LayerNorm(sym_rel_size),
            nn.Linear(sym_rel_size, 1),
        )
        
        self.perpendicular_symbol_to_geo_layer = nn.Sequential(
            nn.Linear(sym_rel_size, sym_rel_size),
            nn.ReLU(),
            nn.LayerNorm(sym_rel_size),
            nn.Linear(sym_rel_size, 1),
        )
        
        self.sigmoid_ac = nn.Sigmoid()
        
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
   
    
    def forward(self, all_geo_info, all_sym_info, all_geo_rels, targets_sym=None):
        
        if self.training:
            assert targets_sym != None
                        
            losses = {
                "text_symbol_geo_rel_loss": all_geo_info[0]["points"].new([0]).zero_().squeeze(),
                "head_symbol_geo_rel_loss": all_geo_info[0]["points"].new([0]).zero_().squeeze(),
                "angle_symbols_geo_rel_loss": all_geo_info[0]["points"].new([0]).zero_().squeeze(),
                "bar_symbols_geo_rel_loss": all_geo_info[0]["points"].new([0]).zero_().squeeze(),
                "parallel_symbols_geo_rel_loss": all_geo_info[0]["points"].new([0]).zero_().squeeze(),
                "perpendicular_symbols_geo_rel_loss": all_geo_info[0]["points"].new([0]).zero_().squeeze(),
            }
            
            for geo_info, sym_info, geo_rel, tar_sym in zip(all_geo_info, all_sym_info, all_geo_rels, targets_sym):
                # construct rel between sym and geo
                sym_to_geo_rel_dict = self.construct_sym_to_geo_per_data(
                    geo_info=geo_info,
                    sym_info=sym_info,
                    geo_rel=geo_rel,
                )
                
                # calculate loss
                per_data_loss = self.cal_sym_geo_rel_loss(
                    per_data_sym_to_geo_rel_dict=sym_to_geo_rel_dict,
                    target=tar_sym,
                )
                
                for sym_geo_key, value in per_data_loss.items():
                    if value != None:
                        if "text_symbol_geo_rel" in sym_geo_key:
                            losses["text_symbol_geo_rel_loss"] += value
                        elif "head_symbol_geo_rel" in sym_geo_key:
                            losses["head_symbol_geo_rel_loss"] += value
                        elif "angle" in sym_geo_key:
                            losses["angle_symbols_geo_rel_loss"] += value
                        elif "bar" in sym_geo_key:
                            losses["bar_symbols_geo_rel_loss"] += value
                        elif "parallel" in sym_geo_key:
                            losses["parallel_symbols_geo_rel_loss"] += value
                        elif "perpendicular" in sym_geo_key:
                            losses["perpendicular_symbols_geo_rel_loss"] += value
                        else:
                            raise ValueError(f"Unknown loss key: ({sym_geo_key})")
                        
            for sym_geo_key, value in losses.items():
                losses[sym_geo_key] = value / len(all_geo_info)
    
            return None, losses
        
        else:
            sym_geo_rels_predictions = []
            for geo_info, sym_info, geo_rel in zip(all_geo_info, all_sym_info, all_geo_rels):

                sym_to_geo_rel_dict = self.construct_sym_to_geo_per_data(
                    geo_info=geo_info,
                    sym_info=sym_info,
                    geo_rel=geo_rel,
                )
                
                # apply sigmoid
                for sym_geo_key, value in sym_to_geo_rel_dict.items():
                    if value != None:
                        sym_to_geo_rel_dict[sym_geo_key] = self.sigmoid_ac(value)

                sym_geo_rels_predictions.append(sym_to_geo_rel_dict)
     
            return sym_geo_rels_predictions, None
            
    def construct_sym_to_geo_per_data(self, geo_info, sym_info, geo_rel):
        """
        Args:
            geo_info (Dict[]):  {"points": Tensor(P, cfg.sym_embed_size) or [], "lines": Tensor(L, cfg.sym_embed_size), "circles": Tensor(C, cfg.sym_embed_size)}
            sym_info (Dict[]): contains symbols information  regarding to different classes, 
                except for the "text_symbols_str" key, other keys' values are in Tensor[?, cfg.sym_embed_size] or [].
            geo_rel (Dict[Tensor(P, ?)]): keys: "pl_rels", "pc_rels", value: Tessor or None.
            
        Returns:
            sym_to_geo_rel_dict (Dict[]): contains the relation score ([0, 1]) between the 
                different symbols and their possible geos.
        """
        
        if len(geo_info["points"]) != 0 and len(geo_info["lines"]) != 0:
            # [P, sym_rel_size]
            LPL_matrix, LPL_mask = self._combine_margin_middle_margin_matrix(
                middle_geo=geo_info["points"],
                margin_geo=geo_info["lines"],
                geo_geo_rel=geo_rel["pl_rels"],
                rel="lpl",
            )
            # [L, sym_rel_size]
            PLP_matrix, PLP_mask = self._combine_margin_middle_margin_matrix(
                middle_geo=geo_info["lines"],
                margin_geo=geo_info["points"],
                geo_geo_rel=geo_rel["pl_rels"].T,
                rel="plp",
            )
        else:
            LPL_matrix = LPL_mask = None
            PLP_matrix = PLP_mask = None
        
        if len(geo_info["circles"]) != 0 and len(geo_info["points"]) != 0:
            # [C, sym_rel_size]
            PCP_matrix, PCP_mask = self._combine_margin_middle_margin_matrix(
                middle_geo=geo_info["circles"],
                margin_geo=geo_info["points"],
                geo_geo_rel=geo_rel["pc_rels"].T,
                rel="pcp",
            )
        else:
            PCP_matrix = PCP_mask = None
        
        contain_head = len(sym_info["head_symbols"]) != 0
        sym_to_geo_rel_dict = {}
        
        """ construct rel between text_symbol and geo """
        # geo_matrix: [P+L+P+L+C, h] or None
        geo_matrix = self.concat_to_matrix(geo_info["points"], geo_info["lines"], LPL_matrix, PLP_matrix, PCP_matrix)
        points_mask = geo_info["points"][0].new([1]).repeat(geo_info["points"].size(0)) if len(geo_info["points"]) != 0 else None
        lines_mask = geo_info["lines"][0].new([1]).repeat(geo_info["lines"].size(0)) if len(geo_info["lines"]) != 0 else None
        # [P+L+P+L+C]
        mask = self.concat_to_matrix(points_mask, lines_mask, LPL_mask, PLP_mask, PCP_mask)
        
        text_symbol_geo_rel = None
        if len(sym_info["text_symbols"]) != 0 and geo_matrix != None:
            # text_symbol_geo_rel: [N, P+L+P+L+C, 1]
            text_symbol_geo_rel = self._construct_rel_between_text_symbols_and_geo(
                symbols=sym_info["text_symbols"],
                geo_matrix=geo_matrix,
                mask=mask,
                symbol_type="text",
                head_symbol=sym_info["head_symbols"] if contain_head else None,
            )
        sym_to_geo_rel_dict["text_symbol_geo_rel"] = text_symbol_geo_rel
 
        """ *********************** End *********************** """
        
        """ construct rel between head_sym and LPL, PLP, PCP """
        head_symbol_geo_rel = None
        if contain_head and geo_matrix != None:
            # geo_matrix: [P+P+L+C, h]
            geo_matrix = self.concat_to_matrix(geo_info["points"], LPL_matrix, PLP_matrix, PCP_matrix)
            # [P+P+L+C]
            mask = self.concat_to_matrix(points_mask, LPL_mask, PLP_mask, PCP_mask)
            # head_symbol_geo_rel: [#head, P+P+L+C]
            head_symbol_geo_rel = self._construct_rel_between_text_symbols_and_geo(
                symbols=sym_info["head_symbols"],
                geo_matrix=geo_matrix,
                mask=mask,
                symbol_type="head",
            )
        sym_to_geo_rel_dict["head_symbol_geo_rel"] = head_symbol_geo_rel  
        
        """ *********************** End *********************** """
        
        """ construct rel between other symbols with geo """
        
        for symbol_name, each_symbol_info in sym_info.items():
            
            if len(each_symbol_info) != 0:  
                
                if "angle" in symbol_name:
                    if each_symbol_info.size(0) > 1 and LPL_matrix != None:     
                        sym_to_geo_rel_dict[f"{symbol_name}_geo_rel"] = self._construct_rel_between_text_symbols_and_geo(
                            symbols=each_symbol_info,
                            geo_matrix=LPL_matrix,
                            mask=LPL_mask,
                            symbol_type="angle")
                        continue
                    
                elif "bar" in symbol_name:
                    if each_symbol_info.size(0) > 1 and PLP_matrix != None:     
                        sym_to_geo_rel_dict[f"{symbol_name}_geo_rel"] = self._construct_rel_between_text_symbols_and_geo(
                            symbols=each_symbol_info,
                            geo_matrix=PLP_matrix,
                            mask=PLP_mask,
                            symbol_type="bar")
                        continue
                    
                elif "parallel" in symbol_name:
                    if each_symbol_info.size(0) > 1 and len(geo_info["lines"]) != 0:     
                        sym_to_geo_rel_dict[f"{symbol_name}_geo_rel"] = self._construct_rel_between_text_symbols_and_geo(
                            symbols=each_symbol_info,
                            geo_matrix=geo_info["lines"],
                            mask=lines_mask,
                            symbol_type="parallel")
                        continue

                elif "perpendicular" in symbol_name:
                    sym_to_geo_rel_dict[f"{symbol_name}_geo_rel"] = self._construct_rel_between_text_symbols_and_geo(
                        symbols=each_symbol_info,
                        geo_matrix=LPL_matrix,
                        mask=LPL_mask,
                        symbol_type="perpendicular")
                    continue
        
        return sym_to_geo_rel_dict
        
    def _construct_rel_between_text_symbols_and_geo(self, symbols, geo_matrix, mask, symbol_type, head_symbol=None):
        """construct relation between text_symbols and P, L, LPL, PLP, PCP, head_symbol(optional).

        Args:
            symbols (Tensor(N, sym_embed_size)) or []
            geo_matrix (Tensor(P+L+P+L+C, h)) or None
            mask (Tensor(P+L+P+L+C)) or None
            head_symbol (Tensor(?, h)): optional or None
        
        Returns:
            text_symbol_geo_rel: [N, P+L+P+L+C, 1]
        """
        
        if head_symbol != None:
            # [P+L+P+L+C+?, h]
            geo_matrix = torch.cat((geo_matrix, head_symbol), dim=0)
            add_mask = mask.new([1]).repeat(head_symbol.size(0))
            # [P+L+P+L+C+?]
            mask = torch.cat((mask, add_mask))
        
        geo_matrix = self.geo_matrix_layer(geo_matrix)
        
        text_symbols_expand = symbols.unsqueeze(1).repeat(1, geo_matrix.size(0), 1)  # [N, P+L+P+L+C, h]
        geo_matrix_expand = geo_matrix.unsqueeze(0).repeat(symbols.size(0), 1, 1)    # [N, P+L+P+L+C, h]
        
        if symbol_type == "text":
            # [N, P+L+P+L+C, h] -> [N, P+L+P+L+C, 1]
            text_symbol_geo_rel = self.text_symbol_to_geo_layer(text_symbols_expand + geo_matrix_expand)
        elif symbol_type == "head":
            text_symbol_geo_rel = self.head_symbol_to_geo_layer(text_symbols_expand + geo_matrix_expand)
        elif symbol_type == "angle":
            text_symbol_geo_rel = self.angle_symbol_to_geo_layer(text_symbols_expand + geo_matrix_expand)
        elif symbol_type == "bar":
            text_symbol_geo_rel = self.bar_symbol_to_geo_layer(text_symbols_expand + geo_matrix_expand)
        elif symbol_type == "parallel":
            text_symbol_geo_rel = self.parallel_symbol_to_geo_layer(text_symbols_expand + geo_matrix_expand)
        elif symbol_type == "perpendicular":
            text_symbol_geo_rel = self.perpendicular_symbol_to_geo_layer(text_symbols_expand + geo_matrix_expand)
        else:
            raise ValueError(f"Unknown symbol type: ({symbol_type})")
        
        # [P+L+P+L+C+?] -> [1, P+L+P+L+C+?] -> [N, P+L+P+L+C+?] -> [N, P+L+P+L+C+?, 1]
        mask_expand = mask.unsqueeze(0).repeat(text_symbol_geo_rel.size(0), 1).unsqueeze(-1)
        
        text_symbol_geo_rel *= mask_expand
        
        return text_symbol_geo_rel

    def cal_sym_geo_rel_loss(self, per_data_sym_to_geo_rel_dict, target):
        """
        Args:
            per_data_sym_to_geo_rel_dict (Dict[]): the keys should match the keys in target.
            target (Dict[]): golden relation between sym and geo.
        """
        
        per_data_loss = {}
        for sym_geo_key in per_data_sym_to_geo_rel_dict.keys():
            
            pred_logits = per_data_sym_to_geo_rel_dict[sym_geo_key]
            try:
                if pred_logits != None:
                    target_ids = target[sym_geo_key]
                    per_data_loss[sym_geo_key] = self.bce_with_logits_loss(input=pred_logits.squeeze(-1), target=target_ids.to(torch.float))
                
                else:
                    per_data_loss[sym_geo_key] = None
            except ValueError:
                
                print("Second: ", seg.get_field("ids"))
                print(pred_logits.size())
                print(target_ids.size())
                raise ValueError
        
        return per_data_loss
    
    def _combine_margin_middle_margin_matrix(self, middle_geo, margin_geo, geo_geo_rel, rel):
        """create matrix to represent relations for LPL, PLP, PCP

        Args:
            middle_geo (Tensor(middle, cfg.geo_embed_siz)): points or lines or circles embeddings.
            margin_geo (Tensor(margin, cfg.geo_embed_siz)): points or lines or circles embeddings.
            geo_geo_rel: (Tensor(middle, margin)): e.g., relation between points and lines (or circles).
        
        Returns:
            geo_geo_matrix (Tensor(middle, h)): relation vector of margin-middle-margin, e.g., LPL.
            middle_qualified_mask (Tensor(middle)): a middle geo is qualified only if it has relations with at least two margin geos.
        """
        
        geo_geo_mask = geo_geo_rel.clone().to(torch.float)
        # endpoint, on_circle
        is_relevant_strong = geo_geo_mask == 1
        # online, center
        is_relevant_weak = geo_geo_mask == 2
        
        # [middle, margin]
        geo_geo_mask[is_relevant_strong] = 1.
        geo_geo_mask[is_relevant_weak] = self.weak_mask
        
        # !!! might cause inf in tensor if enable mix-precision, just set "amp" False will solve this        
        # [middle, margin] * [margin, h] -> [middle, h] concat [middle, h] -> [middle, 2h] -> [middle, h]
        if rel == "lpl":
            geo_geo_matrix = self.LPL_layer(torch.matmul(geo_geo_mask, margin_geo) + middle_geo)
        elif rel == "plp":
            geo_geo_matrix = self.PLP_layer(torch.matmul(geo_geo_mask, margin_geo) + middle_geo)
        elif rel == "pcp":
            geo_geo_matrix = self.PCP_layer(torch.matmul(geo_geo_mask, margin_geo) + middle_geo)
        else:
            raise ValueError(f"Unknown rel type: ({rel})")    
        
        # [middle]
        middle_qualified_mask = (1. - (geo_geo_mask.sum(-1) > 1).float()) * -9999. + 1.
        
        return geo_geo_matrix, middle_qualified_mask.float()

    @staticmethod
    def concat_to_matrix(*args):
        to_concat_list = []
        for vec in args:
            if vec != None:
                if len(vec) != 0:
                    to_concat_list.append(vec)
        if len(to_concat_list) == 0:    # no info available
            return None
        return torch.cat(to_concat_list, dim=0)

class ConstructRel(nn.Module):
    """This module is for train or predict rel over geo and sym information."""
    
    def __init__(self, cfg):
        super(ConstructRel, self).__init__()
    
        self.geo2geo = GeotoGeo(geo_embed_size=cfg.geo_embed_size, geo_rel_size=cfg.geo_rel_size)
        
        self.sym2geo = SymtoGeo(sym_embed_size=cfg.sym_embed_size, sym_rel_size=cfg.sym_rel_size, weak_mask=cfg.weak_mask)
    
    def forward(self, all_geo_info, all_sym_info, targets_geo=None, targets_sym=None):
        """
        Args:
            - all_geo_info List[Dict]: Contain batch data geo information, 
                each dict contains geo information regarding to different classes, in Tensor([N, cfg.geo_embed_size])
                
            - all_sym_info List[Dict]: Contain batch data symbols information, each dict contains symbols information 
                regarding to different classes, except for the "text_symbols_str" key, 
                other keys' values are in Tensor[?, cfg.sym_embed_size].
                
            - targets_geo (List[Dict], optional): The golden relation between (points and lines), (points and circles),
                (points and lines, pl_rels): [P, L], 0: no-rel, 1: endpoint, 2: online
                (points and circles, pc_rels): [P, C], 0: no-rel, 1: center, 2: oncircle
            
            - targets_sym (List[Dict]): The golden relation between:
                (P: point L: line LPL: line-point-line PLP: point-line-point PCP: point-circle-point H: head)
                
                1. text_symbol_geo_rel: text_symbols and (P, L, LPL, PLP, PCP, H): [ts, P+L+LPL+PLP+PCP+H], 0: no-rel, 1: rel
                
                2. head_symbol_geo_rel: (optional) head and (LPL, PLP, PCP): [H, P+L+C], 0: no-rel, 1: rel
                
                3. [None|double_|triple_|quad_|penta_]angle_symbols_geo_rel: angle_symbols (including xxx_angle) and (LPL): [as, P], 0: no-rel, 1: rel
                
                4. [None|double_|triple_|quad_]bar_symbols_geo_rel: bar_symbols (including xxx_bar) and (PLP): [bs, L], 0: no-rel, 1: rel
                
                5. [None|double_|parallel_]parallel_symbols_geo_rel: parallel_symbols (including xxx_parallel) and (L): [ps, L], 0: no-rel, 1: rel
                
                6. perpendicular_symbols_geo_rel: perpendicular_symbols: [pps, P], 0: no-rel, 1: rel
        """
        
        if self.training:
            assert targets_geo != None and targets_sym != None    
        
        # # # # # # # # # Geo2Geo Relation Build # # # # # # # # #
        
        # geo_rels_predictions: [{"pl_rels": List[Tensor(P, L)], "pc_rels": List[Tensor(P, C)]}] len==bsz
        #                       already use argmax select from [0, 1, 2].
        # geo_rel_losses: {"pl_loss": Tensor, "pc_loss": Tensor}

        geo_rels_predictions, geo_rel_losses = self.geo2geo(all_geo_info, targets_geo)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        
        
        # # # # # # # # # Sym2Geo Relation Build # # # # # # # # #
        
        # sym_geo_rels_predictions: List[Dict], len==bsz, each dict contain different sym2geo rel as key.
        #                           just return the prob after sigmoid.
        # sym_geo_rel_losses: Dict[], keys are different sym2geo rel.
        sym_geo_rels_predictions, sym_geo_rel_losses = self.sym2geo(
            all_geo_info=all_geo_info,
            all_sym_info=all_sym_info,
            all_geo_rels=targets_geo if self.training else geo_rels_predictions,
            targets_sym=targets_sym,
        )
    
        # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
    
        if self.training:
            losses = {}
            losses.update(geo_rel_losses)
            losses.update(sym_geo_rel_losses)
            
            return geo_rels_predictions, sym_geo_rels_predictions, losses

        else:
            
            return geo_rels_predictions, sym_geo_rels_predictions, None