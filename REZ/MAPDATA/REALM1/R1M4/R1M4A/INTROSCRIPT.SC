Name "introscript"

AbortOnDamage FALSE
AbortOnEnemySighted FALSE
TimeOut 999.0
SightRange	1200
FieldOfView	 360
FaceMoveDir TRUE
Walkspeed 100
YawRate 360
CheckForEnemyInterval	0.5			
TargetPlayer	
		
OnEnemySighted_Goto		Attack
OnDamage_Goto			FarAttack

StartAnim	"AmbA"
Wait		0.5
						
StartAnim	"Walk"
TurnTo		itarget1
MoveTo		itarget1		
StartAnim	"AmbA"
Wait		2

Walkspeed 1500

StartAnim	"Run"
TurnTo		itarget2
MoveTo		itarget2	
Walkspeed 500	
TurnTo		itarget3
MoveTo		itarget3
TurnTo		itarget4
MoveTo		itarget4
TurnTo		itarget5
MoveTo		itarget5
TurnTo		itarget6
MoveTo		itarget6

StartAnim	"AmbA"
Wait		2.8
TurnToAngle	303
StartAnim	"AttackA"
Wait		1.0
StartAnim	"AmbA"
Wait		1.0

Attack:

        OnEnemySighted_Goto		DISABLE
	OnDamage_Goto			DISABLE
	OnEnemyOutOfRange_Goto	        DISABLE
	YawRate		360.0
	TurnToTarget	0.57	
	RangedAttackTarget
        MeleeAttackTarget
FarAttack:
	If_DamageFromPlayer_Goto	BoostSightRange
	
BoostSightRange:
SightRange	2048.0
	